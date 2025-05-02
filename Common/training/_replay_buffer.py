"""
Functions to set up and update the replay buffer. It is assumed that the main
function 'manage_replay_buffer' is run as a separate process from the main
process. Communication happens through multiprocessing queues.
"""
import time
import queue
import jax
import jax.numpy as jnp
import numpy as np
import contextlib
import multiprocessing

from training._utils import initializer_cpu_only, get_init_apply
from training._utils import mean_var_update, mean_var_finalize
import interface.sample_creation
import training.value_functions




def initializer_sample_creation(settings_):  # in sample_creation_pool
    global apply, settings
    settings = settings_

    backend = settings['sample_creation']['backend']

    if backend == 'cpu':
        initializer_cpu_only()

    _, apply = get_init_apply(settings, backend)
    training.value_functions.apply = apply


def create_sample(q_samples, seed, params, state):  # in sample_creation_pool
    sample = interface.sample_creation.create_sample(
        params, state, apply, settings, seed=seed)

    q_samples.put(sample)


def add_new_samples_to_replay_buffer_mixed(q=None):  # runs in sample manager process
    samples = q_samples.get()
    next_idx = rb['start_idx'] + rb['n_samples']

    for sid, sample in enumerate(samples):
        rb[next_idx] = sample
        rb['times'].append(time.time())
        rb['n_samples'] += 1
        rb['size'] += len(sample[0])
        next_idx += 1

        # inform main process that a new game has been added to the rb
        # if q is not None and sid % 2 == 1:
            # q.put('new_sample')


def add_new_samples_to_replay_buffer_sequential(q=None):  # runs in sample manager process
    samples = q_samples.get()
    len_samples = sum([len(s[0]) for s in samples])
    
    # if all samples (except for one) have been used up then add the new
    # samples (even if their size exceeds the target replay buffer)
    while rb['target_size'] - rb['size'] < len_samples and rb['n_samples'] > 2:
        trim_replay_buffer_last()
        time.sleep(1e-3)

        if rb.get('stop', False) is True:
            return

    next_idx = int(rb['start_idx'] + rb['n_samples'])

    for sid, sample in enumerate(samples):
        rb[next_idx] = sample
        rb['times'].append(time.time())
        rb['n_samples'] += 1
        rb['size'] += len(sample[0])
        next_idx += 1

        # inform main process that a new game has been added to the rb
        # if q is not None and sid % 2 == 1:
            # q.put('new_sample')


def add_new_samples_to_replay_buffer(q=None):  # runs in sample manager process
    if settings['replay_buffer']['sequential_sampling'] is True and settings['replay_buffer'].get('remove_last', False) is True:
        return add_new_samples_to_replay_buffer_sequential(q)

    return add_new_samples_to_replay_buffer_mixed(q)


def initialize_replay_buffer(ss):  # runs in sample manager process
    n_workers = settings['sample_creation']['n_sample_creators']
    initial_rb_size = settings['replay_buffer']['initial_replay_buffer_size']

    n_jobs = 0
    first_sample = True

    while True:
        key = ss.spawn(1)[0]
        ar = sample_creation_pool.apply_async(
            create_sample, (q_samples, key, params, state))

        # incorrect training settings can cause errors in the pool, these
        # errors are only reraised in the main thread if we 'get' the result,
        # do that once here to inform the user about errors
        if first_sample is True:
            first_sample = False
            ar.get()

        if n_jobs < n_workers:
            n_jobs += 1

        else:  # a worker has created a sample, add it to rb
            add_new_samples_to_replay_buffer_mixed()
            print(f'Replay Buffer Size: {rb["size"]}', end='        \r')

            if rb['size'] >= initial_rb_size:
                break


def get_target_rb_size_single_game(target_rb_size):
    # there are two samples per game --> only keep the last two samples
    if rb['n_samples'] < 2:
        # if target >= rb['size'] then no samples will be removed
        return rb['size']

    idx0 = (rb['start_idx'] + rb['n_samples'] - 2) % rb['max_idx']
    idx1 = (idx0 + 1) % rb['max_idx']
    return len(rb[idx0][0]) + len(rb[idx1][0])


def remove_sample_from_rb(idx):
    rb['size'] -= len(rb[idx][0])
    rb['n_samples'] -= 1
    del rb[idx]


def get_mean_var_agg():
    return (
        rb.get('stats_mean', 0),
        rb.get('stats_var', 0),
        rb.get('stats_agg', 0))


def update_mean_var_agg(t):
    mean_var_agg = get_mean_var_agg()
    mean_var_agg = mean_var_update(mean_var_agg, t)
    set_mean_var_agg(mean_var_agg)


def finalize_mean_var_agg():
    mean_var_agg = get_mean_var_agg()
    return mean_var_finalize(mean_var_agg)


def set_mean_var_agg(mean_var_agg):
    rb['stats_mean'] = mean_var_agg[0]
    rb['stats_var'] = mean_var_agg[1]
    rb['stats_agg'] = mean_var_agg[2]


# Todo: rename to more appropriate name
def trim_replay_buffer_last():  # this can be used in 'sequential' mode
    if 'next_idx' not in rb:
        return

    # remove already used samples
    while rb['start_idx'] < rb['next_idx'] and rb['n_samples'] > 2:
        update_mean_var_agg(time.time() - rb['times'][0])
        del rb['times'][0]

        remove_sample_from_rb(rb['start_idx'])
        rb['start_idx'] += 1

    print(f'\nStart={rb["start_idx"]}, Next={rb["next_idx"]}, N_Samples={rb["n_samples"]}, Size={rb["size"]}, Target_Size={rb["target_size"]}, Time={time.time():.2f}     \033[F', end='')


def trim_replay_buffer_first():
    while rb['size'] > rb['target_size'] and rb['n_samples'] > 1:
        update_mean_var_agg(time.time() - rb['times'][0])
        del rb['times'][0]

        remove_sample_from_rb(rb['start_idx'])
        rb['start_idx'] += 1


def trim_replay_buffer():
    if settings['replay_buffer']['sequential_sampling'] is True and settings['replay_buffer'].get('remove_last', False) is True:
        return trim_replay_buffer_last()

    return trim_replay_buffer_first()


def run_replay_buffer(ss):
    global params, state

    n_workers = settings['sample_creation']['n_sample_creators']
    rb['target_size'] = settings['replay_buffer']['target_replay_buffer_size']

    only_one_game_in_rb = \
        settings['replay_buffer']['only_one_game_in_replay_buffer']

    while True:
        # for testing purposes we can require the sample creator to wait with
        # adding a new sample to the rb until the main process has used the
        # previous sample (this is like training without a rb); otherwise we
        # just check for the 'stop' signal
        msg = 'ready'

        if only_one_game_in_rb is True or q_comm_in.qsize() > 0:
            msg = q_comm_in.get()

        if msg == 'stop':
            rb['stop'] = True  # need to tell sequential mode to stop
            print('Stopped sample creation for replay buffer.')
            return

        assert msg == 'ready', f'message was {msg}'  # wait until main process is ready for new sample
        # print('\n\nrb continuing, samples waiting:', q_samples.qsize(), '\n')

        # retrieve previously created sample and update replay buffer
        add_new_samples_to_replay_buffer()

        # make sure we use up-to-date params
        while q_params.qsize() > 0:
            with contextlib.suppress(queue.Empty):  # main process may remove old params when updating queue with new params
                params, state = q_params.get()

        # start creation of new sample
        key = ss.spawn(1)[0]

        sample_creation_pool.apply_async(
            create_sample, (q_samples, key, params, state))

        # keep size of replay buffer roughly constant;
        if only_one_game_in_rb:
            rb['target_size'] = get_target_rb_size_single_game(target_rb_size)

        trim_replay_buffer()

        mean, var = finalize_mean_var_agg()[:2]
        print(f'\033[Fsample times: {mean:.2f}s (std={var**0.5:.2f}s)'
              + f'  qsize={q_samples.qsize()}'
              + ' ' * 20 )

        # inform main process that a new sample has been added to the rb
        q_comm_out.put('new_sample')


def run(settings_, rb_, q_samples_, q_params_, q_comm_out_, q_comm_in_):
    global params, state, settings
    global sample_creation_pool, q_samples, q_params, q_comm_out, q_comm_in, rb
    print('manage_replay_buffer')

    settings = settings_
    q_samples, q_params, rb = q_samples_, q_params_, rb_
    q_comm_out, q_comm_in = q_comm_out_, q_comm_in_

    # setup replay buffer
    rb['start_idx'] = 0
    rb['n_samples'] = 0
    rb['max_idx'] = 1e9  # some number much larger than the length of one sample
    rb['size'] = 0  # total number of moves in replay buffer

    # get initial parameters sent from main process
    params, state = q_params.get()

    # pool is used to create samples
    sample_creation_pool = multiprocessing.Pool(
        processes=settings['sample_creation']['n_sample_creators'],
        initializer=initializer_sample_creation,
        initargs=(settings,))

    ss = np.random.SeedSequence(settings['sample_creation']['seed'])
    initialize_replay_buffer(ss.spawn(1)[0])

    # tell main process that replay buffer is initialized
    q_comm_out.put('rb_ready')
    print('Finished initializing replay buffer.\n')

    # continuously update replay buffer until told to stop
    run_replay_buffer(ss)

    # clean up
    sample_creation_pool.close()
    sample_creation_pool.join()
