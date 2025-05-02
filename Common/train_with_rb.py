import jax
import jax.numpy as jnp
import optax
import numpy as np
import multiprocessing
import multiprocessing.managers

# this file will be symlinked in the respective game-folders, running a
# symlink'ed file uses the path of the target file, not of the link file
import os
import sys
sys.path.remove(sys.path[0])
sys.path.append(os.path.dirname(__file__))

import time
import datetime

import atexit
import signal

import training._utils, training._replay_buffer
import training.loss_functions, interface.sample_creation
import training.target_functions_sampling
import training.value_functions



# if something goes wrong or if you interrupt training then you may not
# want to keep the output folder, the exit handler provides that option
def exit_handler(settings):
    """ Offer option to remove output folder on abnormal exit. """
    def fn():
        d = input('\n\nFinished, do you want to keep the output folder? (y/n):\n')
        print('\n')

        if d == 'n':
            save_path = settings['workings']['save_path']
            print(f'Deleting folder {save_path}.')

            import shutil
            shutil.rmtree(save_path)

            print('Deleted. Note, if evaluation was running then it may be'
                  ' recreated and contain the evaluation result.')

    return fn


def initializer(settings, load_optimizer=True):
    """
    Initialize 'apply' so it's a global variable in each process and only
    gets compiled once.
    """
    global init
    global apply
    global optimizer

    backend = settings['training']['backend']

    jax.config.update('jax_platform_name', backend)
    jax.default_device = jax.devices(backend)[0]

    init, apply = training._utils.get_init_apply(settings, backend, jit_compile=False)
    training._utils.init = init
    training.loss_functions.apply = apply  # will be used in gradient, which will be compiled separately (I remember from somewhere that compiling a compiled function is not ideal, Todo: double check that)

    _, apply = training._utils.get_init_apply(settings, backend, jit_compile=True)
    training.value_functions.apply = apply

    if load_optimizer:
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
    while True:
        try:
            q_params.get(block=False)
        except:
            break


def send_params_to_players(q_params, params, state):
    empty_queue(q_params)
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

    # if something goes wrong or if you interrupt training then you may not
    # want to keep the output folder, the exit handler provides that option
    exit_fn = exit_handler(settings)
    atexit.register(exit_fn)
    signal.signal(signal.SIGINT, exit_fn)
    signal.signal(signal.SIGTERM, exit_fn)

    # set up shared objects 
    m = multiprocessing.managers.SyncManager()
    m.start(initializer=training._utils.initializer_cpu_only)
    q_samples, q_params = m.Queue(), m.Queue()
    q_comm_out, q_comm_in = m.Queue(), m.Queue()

    rb = m.dict()  # replay buffer
    rb['indices'] = m.list()
    rb['times'] = m.list()  # keep track of time a sample is in replay buffer

    """  using the pool is slower than without it
    # if multiple sequences are used in a sample then the target has to be
    # computed for each sequence, this is slow if done sequentially so use a
    # pool to parallelize it (ideally max_n_sequences_per_sample is a multiple
    # of n_sampling processes)
    sampling_pool = multiprocessing.Pool(
        processes=settings['replay_buffer']['n_sampling_processes'],
        initializer=initializer, initargs=(settings, False))

    interface.sample_creation.sampling_pool = sampling_pool
    """

    # make parameters available to sample creators
    q_params.put((params, state))

    # start populating replay buffer
    rb_process = multiprocessing.Process(
        target=training._replay_buffer.run,
        args=(settings, rb, q_samples, q_params, q_comm_in, q_comm_out))

    # rb_process.run()  # for testing
    rb_process.start()

    # set up functions required for training
    loss_fn = training.loss_functions.get_loss_fn(settings)
    grad_fn = jax.jit(jax.grad(loss_fn))
    # grad_fn = jax.grad(loss_fn)

    epoch = settings['workings']['epoch']
    n_epochs_since_update = 0

    only_one_game_in_rb = \
        settings['replay_buffer']['only_one_game_in_replay_buffer']

    max_epochs_per_game = \
        settings['replay_buffer']['epochs_per_game']

    # if we sample sequentially from the replay buffer then the sample creator
    # waits until a new sample becomes available (don't need to check here)
    if settings['replay_buffer']['sequential_sampling'] is True:
        max_epochs_per_game = float('inf')

    # wait until replay buffer is initialized
    assert (msg := q_comm_in.get()) == 'rb_ready', \
        f'Initialization of replay buffer returned unexpected value {msg}.'

    # training loop
    while epoch != (max_epochs := settings['training']['max_epochs']):
        training._utils.measure_speed_start(settings, epoch)

        # sample: (board, options, idx, R)
        st = time.time()
        sample = interface.sample_creation.sample_from_rb(rb, settings, params, state)  # Todo: separate process pool to pre-load samples?
        et = time.time()

        # visualize activations to get evaluate adequacy of model
        def visualize(state):
            import matplotlib.pyplot as plt
            acts = state['~']['activations']
            plt.plot(acts[:, 0], label='mean',  marker='o')
            plt.plot(acts[:, 1], label='var',  marker='o')
            plt.axhline(0, linestyle='--', color='gray')
            plt.legend()
            plt.ylim(-3, 100)
            save_path = os.path.join(settings['workings']['save_path'], f'activations_{epoch}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        if epoch % 1000 == 0:
            (Q, _), state = apply(params, state, sample[0], p=True)
            visualize(state)

        # loss = loss_fn(params, state, sample[0], sample[2], sample[3], options=sample[1])
        # print(f'loss {epoch}/{max_epochs}: {loss[0]:.4f}', end='               \r')

        grads = grad_fn(params, state,
                       inputs=sample[0],
                       idx=sample[2],
                       target=sample[3],
                       options=sample[1])

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
        print(f'epoch {epoch}/{max_epochs} ({epoch/max_epochs*100:.1f}%) sampling={et-st:.2e}s ({(et-st)/(et_-st):.2f})',
              end='               \r')

        # for testing purposes we can require the sample creator to wait with
        # adding a new sample to the rb until the main process has used the
        # previous sample
        if only_one_game_in_rb is True:
            q_comm_out.put('ready')

        n_epochs_since_update += 1

        if q_comm_in.qsize() > 0 or only_one_game_in_rb  or n_epochs_since_update == max_epochs_per_game:
            msg = q_comm_in.get()
            empty_queue(q_comm_in)
            n_epochs_since_update = 0
            # print('\nnew sample\n')

        training._utils.measure_speed_end(settings, epoch)

    print('Sending signal to stop sample creation.')
    q_comm_out.put('stop')

    # save final parameters
    training._utils.save(
        settings, dict(params=params, state=state, opt_state=opt_state))

    # finish up
    rb_process.join()
    print(f'Training finished after {time.time()-start_} seconds.')
