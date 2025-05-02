"""
Training process where samples are created in a multiprocessing pool and
gradients are computed in the pool. The main process only applies the
gradients. Each game is used as a minibatch, ie all samples in a minibatch
are highly dependent. After the sample has been used it will be discarded,
ie each sample is only used once.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
import multiprocessing
import multiprocessing.managers

import training._utils, training._replay_buffer
import training.loss_functions, interface.sample_creation
import interface.sample_creation
import training.value_functions
import training.target_functions_sampling
from training.value_functions import get_value_fn

import os
import sys
import time
import queue  # for queue.Empty exception
import datetime
import contextlib



def contains_nan(pytree):
    for l in jax.tree_util.tree_leaves(pytree):
        if jnp.isnan(l).any():
            return True

    return False


# -----------------------------------------------------------------------------
#  Sample Creation Process
# -----------------------------------------------------------------------------
def initializer_sample_creation(settings_):  # in sample_creation_pool
    global apply, settings, grad_fn
    settings = settings_

    backend = settings['sample_creation']['backend']

    if backend == 'cpu':
        training._utils.initializer_cpu_only()

    _, apply = training._utils.get_init_apply(settings, backend)
    training.value_functions.apply = apply
    training.loss_functions.apply = apply

    loss_fn = training.loss_functions.get_loss_fn(settings)
    # grad_fn = jax.jit(jax.grad(loss_fn))
    grad_fn = jax.grad(loss_fn)


def process_sample(samples, q_grads, seed, params, state):  # in sample_creation_pool
    value_fn = get_value_fn(settings)
    target_fn = training.target_functions_sampling.get_target_fn(settings)

    # use fixed batch size to avoid repeated compilation of jax code
    bs = settings['training']['batch_size']

    inputs = -np.ones((bs,) + samples[0][0].shape[1:])
    options = -np.ones((bs,) + samples[0][1].shape[1:], dtype=int)
    idx = -np.ones((bs,) + samples[0][2].shape[1:], dtype=int)
    targets = np.zeros((bs,) + samples[0][3].shape[1:])

    # pick indices to use in training
    moves = [(sid, i)
             for sid, sample in enumerate(samples)
             for i in range(len(sample[0]))]

    n_moves = len(moves)
    n_indices = min(n_moves, bs)

    rng = np.random.default_rng(seed=seed)  # reusing seed, not ideal
    indices = rng.choice(moves, size=n_indices, replace=False)

    sids = np.array([s[0] for s in indices])
    mids = np.array([s[1] for s in indices])
    used_sample_ids = np.unique(sids)

    # process samples
    start = end = 0

    for sid in used_sample_ids:
        sample = samples[sid]
        move_ids = mids[sids == sid]
        end = start + len(move_ids)

        inputs[start:end] = sample[0][move_ids]
        options[start:end] = sample[1][move_ids]
        idx[start:end] = sample[2][move_ids]

        target = np.asarray(
            target_fn(sample, settings=settings, val_fn=value_fn,
                      params=params, state=state))

        targets[start:end] = target[move_ids]
        start = end

    # compute gradients
    grads = grad_fn(params, state,
                   inputs=inputs,
                   idx=idx,
                   target=targets,
                   options=options)

    if q_grads.qsize() > settings['no_replay_buffer']['max_queue_size']:
        q_grads.get()

    q_grads.put(grads)


def create_sample(q_grads, seed, params, state):  # in sample_creation_pool
    # get sample
    samples = interface.sample_creation.create_sample(
        params, state, apply, settings, seed=seed)

    n_samples = len(samples)

    # there are two samples per game, previously I only created one game at a
    # time and used the entire game (ie two sequences of moves) for training,
    # this is no longer necessary, you can choose in the settings how many
    # sequences should be used
    n_seq = settings['no_replay_buffer']['n_sequences_per_update']

    for i in range(n_samples // n_seq):
        process_sample(samples[n_seq*i:n_seq*(i+1)],
                       q_grads, seed, params, state)


def wait_for_available_sample_creator(async_res, n_creators):
    if len(async_res) < n_creators:
        return async_res

    while True:
        for ar in async_res:
            if ar.ready():
                ar.get()  # reraises errors if there are any
                async_res.remove(ar)
                return async_res

        time.sleep(1e-3)


def run_sample_creation(settings, q_grads, q_params, q_comm):  # in separate process
    n_creators = settings['sample_creation']['n_sample_creators']

    # get initial parameters sent from main process
    params, state = q_params.get()

    # pool is used to create samples
    sample_creation_pool = multiprocessing.Pool(
        processes=n_creators,
        initializer=initializer_sample_creation,
        initargs=(settings,))

    ss = np.random.SeedSequence(settings['sample_creation']['seed'])
    async_res = list()

    while True:
        # check for message from main process
        while q_comm.qsize() > 0:
            if (msg := q_comm.get()) == 'stop':
                # clean up
                sample_creation_pool.close()
                sample_creation_pool.join()

                print('Stopped sample creation for replay buffer.')
                return

        # check if any of the creators is ready to start creating a new sample
        async_res = wait_for_available_sample_creator(async_res, n_creators)

        # make sure we use up-to-date params
        while q_params.qsize() > 0:
            with contextlib.suppress(queue.Empty):  # main process may remove old params when updating queue with new params
                params, state = q_params.get()

        # start creation of new sample
        key = ss.spawn(1)[0]

        ar = sample_creation_pool.apply_async(
            create_sample, (q_grads, key, params, state))

        async_res.append(ar)


# -----------------------------------------------------------------------------
#  Main Process
# -----------------------------------------------------------------------------
def initializer(settings):
    """
    Initialize 'apply' so it's a global variable in each process and only
    gets compiled once.
    """
    global init
    global optimizer

    backend = settings['training']['backend']

    jax.config.update('jax_platform_name', backend)
    jax.default_device = jax.devices(backend)[0]

    init, _ = training._utils.get_init_apply(settings, backend)
    training._utils.init = init

    # (dynamically) load optimizer
    sys.path.append(settings['workings']['settings_folder'])
    import optimizer_settings
    optimizer = optimizer_settings.load_optimizer(settings)


@jax.jit
def update(grads, params, opt_state):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def empty_queue(q):
    # Todo: maybe always leave one previous set of params in queue (to ensure that the main process doesn't delete faster than sample creators can get)
    while True:
        try:
            q_params.get(block=False)
        except:
            break


def send_params_to_players(q_params, params, state):
    empty_queue(q_params)  # remove any pending (old) parameters before putting new ones
    q_params.put((params, state))


if __name__ == '__main__':
    start_ = time.time()

    # required for JAX to work with multiprocessing
    multiprocessing.set_start_method('spawn')

    # load settings and params
    args = training._utils.parse_arguments()
    settings = training._utils.load_settings_file(args)
    initializer(settings)

    settings['workings']['rng'] =  \
        np.random.default_rng(settings['training']['seed'])

    if args.continue_training is None:
        params, state = training._utils.load_new_params(settings)
        opt_state = optimizer.init(params)

    else:
        params, state, opt_state = \
            training._utils.load_existing_params(
                *args.continue_training, settings=settings)

        if args.same_folder is True:
            settings['workings']['save_path'] = args.continue_training[0]
            settings['workings']['copied_settings'] = True

    training._utils.copy_settings_to_output_folder(settings)

    # set up shared objects 
    m = multiprocessing.managers.SyncManager()
    m.start(initializer=training._utils.initializer_cpu_only)
    q_grads, q_params, q_comm = m.Queue(), m.Queue(), m.Queue()

    # make parameters available to sample creators
    q_params.put((params, state))

    # start creating samples
    sc_process = multiprocessing.Process(
        target=run_sample_creation,
        args=(settings, q_grads, q_params, q_comm))

    # sc_process.run()  # for testing
    sc_process.start()

    epoch = settings['workings']['epoch']

    # training loop
    while epoch != (max_epochs := settings['training']['max_epochs']):
        training._utils.measure_speed_start(settings, epoch)

        # sample: (board, options, idx, R)
        st = time.time()
        grads = q_grads.get()
        et = time.time()
        assert not contains_nan(grads), 'Grads contain NaN'

        params, opt_state = update(grads, params, opt_state)

        if epoch % settings['sample_creation']['param_update_interval'] == 0:
            send_params_to_players(q_params, params, state)

        training._utils.check_for_manual_changes(settings)

        epoch += 1
        settings['workings']['epoch'] = epoch

        training._utils.save(
            settings, dict(params=params, state=state, opt_state=opt_state))

        if settings['workings'].get('STOP', False):
            print('\nManual Stopping'); break


        et_ = time.time()
        print(f'epoch {epoch}/{max_epochs} ({epoch/max_epochs*100:.1f}%) sampling={et-st:.2e}s ({(et-st)/(et_-st):.2f}) queue={q_grads.qsize()}',
              end='               \r')

        training._utils.measure_speed_end(settings, epoch)


    print('Sending signal to stop sample creation.')
    q_comm.put('stop')

    # save final parameters
    training._utils.save(
        settings,
        dict(params=params, state=state, opt_state=opt_state),
        check_save_interval=False)

    # finish up
    sc_process.join()
    print(f'Training finished after {time.time()-start_} seconds.')
