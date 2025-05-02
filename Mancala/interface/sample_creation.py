"""
Todo: the only problem-specific code is process_games; the average number
length of a game is required in update_sample_use_count but that could be
read from settings!

Create a sample by playing the game with given neural network parameters and
converting the output to a format useful for reinforcement learning. 
"""
import os
import sys
import time
import numpy as np
import jax.numpy as jnp 
import contextlib

from simulator.machine_game import flip_board, flip_indices
from simulator import machine_game
from interface import machine_agent
from training.target_functions_creation import get_target_fn
import training.target_functions_sampling
import training.value_functions
from training.value_functions import get_value_fn
from training._policies import random_policy, get_eps_greedy_policy
from training._policies import get_policy


# -----------------------------------------------------------------------------
#  Functions for Creating Samples
# -----------------------------------------------------------------------------
def process_games(samples, settings, params, state):
    target_fn = get_target_fn(settings)
    value_fn = get_value_fn(settings)

    aux = dict()
    _samples = list()

    board, options, idx, winner = samples
    # print('\nshapes:', len(board), len(options), len(idx), len(winner))
    n_sequences = len(board)

    for i in range(n_sequences):
        # [::-1] to have last move first, which is used by the remaining code
        _b = np.array(board[i])[::-1]
        _o = np.array(options[i])[::-1]
        _i = np.array(idx[i])[::-1]
        # print(f'game {i} shapes: board={_b.shape}, options={_o.shape}, idx={_i.shape}')
        current_player = i % 2
        # print('current player:', current_player)
        # print('options -1:', _o[_i == -1])

        # flip view from player 1 to player 0 (training always takes place
        # from view of player 0)
        if current_player == 1:
            _b = flip_board(_b)
            _o = flip_indices(_o)
            _i = flip_indices(_i)

        if winner[i] == 0.5:
            R = 0

        else:
            R = 1.0 if (winner[i] == current_player) else -1.0

        target, aux = target_fn(  # target_fn is for individual samples, not for entire games!
            (_b, _o, _i, R), aux,
            settings=settings, val_fn=value_fn, params=params, state=state)

        # print(f'\n\nsample {i}: _b={_b.shape}, _o={_o.shape}, _i={_i.shape}, target={target.shape}\n\n')

        check_for_nan(_b, 'board')
        check_for_nan(_o, 'options')
        check_for_nan(_i, 'idx')
        check_for_nan(target, 'target')

        _samples.append((_b, _o, _i, target))

        # """
        if settings['workings'].get('p1', False) is False and (i == 0 or i == 1):
            print('\n _b:', _b)
            print(' _o:', _o)
            print(' _i:', _i)
            print(' R:', R)
            print(' target:', target)

        if settings['workings'].get('p1', False) is False and i == 1:
            settings['workings']['p1'] = True
        # """

    # raise RuntimeError('...')
    return _samples


def create_sample(params, state, apply, settings, seed=None):  # called in sample creation pool
    if seed is None:
        seed = int(time.time())

    # (dynamically) load training_policies
    sys.path.append(settings['workings']['settings_folder'])
    from training_policies import policies

    n_policies = len(policies)
    rng = np.random.default_rng(seed=seed)
    machine_players = list()

    idx_0 = rng.integers(low=0, high=n_policies)
    policy = get_policy(idx_0, settings)
    mp = machine_agent.create(params, state, policy, apply)
    machine_players.append(mp)

    idx_1 = rng.integers(low=0, high=n_policies)

    if idx_1 != idx_0:
        policy = get_policy(idx_1, settings)
        mp = machine_agent.create(params, state, policy, apply)

    machine_players.append(mp)

    samples = machine_game.play(machine_players, seed=seed, settings=settings)
    samples = process_games(samples, settings, params, state)

    return samples


# -----------------------------------------------------------------------------
#  Functions for Sampling from Replay Buffer
# -----------------------------------------------------------------------------
def check_for_nan(l, name):
    assert not jnp.isnan(l).any(), f'NaNs in {name}, {l}'


def update_sample_use_count(settings, rb, used_sample_ids, sids, mids):
    if 'sample_use' not in settings['workings']:
        settings['workings']['sample_use'] = dict()
        settings['workings']['sample_use']['first_idx'] = 0
        settings['workings']['sample_use']['last_saved_idx'] = 0

    for sid in used_sample_ids:
        move_ids = mids[sids == sid]

        if sid not in settings['workings']['sample_use']:
            # sample may have been deleted between retrieval and this function,
            try:  # assume average length of 25
                settings['workings']['sample_use'][sid] = \
                    np.zeros(len(rb[sid][0]), dtype=int)

            except KeyError:
                settings['workings']['sample_use'][sid] = \
                    np.zeros(max(move_ids.max()+1, 25), dtype=int)

        settings['workings']['sample_use'][sid][move_ids] += 1

    while True:
        if settings['workings']['sample_use']['first_idx'] in rb:
            break

        settings['workings']['sample_use']['first_idx'] += 1

    first_idx = settings['workings']['sample_use']['first_idx']
    last_saved_idx = settings['workings']['sample_use']['last_saved_idx']

    if first_idx - last_saved_idx > 500:  # only write to file every 500 updates (faster than at every update and less memory than writing at end)
        save_path = settings['workings']['save_path']
        save_path = os.path.join(save_path, 'sample_use.txt')

        with open(save_path, 'a') as f:
            for sid in range(last_saved_idx, first_idx):
                if sid not in settings['workings']['sample_use']:
                    # I do want to report how many samples went unused
                    f.write('0\n')
                    continue

                f.write(str(settings['workings']['sample_use'][sid])[1:-1] + '\n')
                del settings['workings']['sample_use'][sid]

        settings['workings']['sample_use']['last_saved_idx'] = first_idx - 1


def sample_from_rb_mixed(rb, settings, params, state):  # called in main process
    """ Draw random samples from the replay buffer. """
    bs = settings['training']['batch_size']
    rng = settings['workings']['rng']
    count = 0

    # not using all indices to better deal with concurrent deletions
    rb_offset = settings['replay_buffer']['replay_buffer_sampling_offset']

    if settings['replay_buffer']['only_one_game_in_replay_buffer'] is True:
        rb_offset = 0

    value_fn = get_value_fn(settings)
    target_fn = training.target_functions_sampling.get_target_fn(settings)

    # set up arrays to hold samples (-1 will never occur in game, used in loss
    # functions to identify padding)
    mid_idx = int(rb['start_idx'] + rb['n_samples']//2)
    s = rb[mid_idx]

    inputs = -1 * np.ones((bs,) + s[0].shape[1:])
    options = -1 * np.ones((bs,) + s[1].shape[1:], dtype=int)
    idx = -1 * np.ones((bs,), dtype=int)
    targets = -1 * np.ones((bs,))

    # size of rb may change while we sample, if we couldn't get a sample
    # --> try again
    while True:
        # with contextlib.suppress(KeyError):
        try:
            # rb_target_size is given in number of moves (if it was given in
            # number of samples then we couldn't ensure a maximum memory usage
            # b/c different samples have different numbers of moves); so
            # rb_offset is also given in number of moves (it would be confusing
            # for the user if it was given in number of samples/games); here we
            # increase the start sample index until the desired number of moves
            # is reached
            start, c = rb['start_idx'], 0

            while c < rb_offset:
                c += len(rb[start][0])
                start += 1

            # pick indices to use in training
            end = rb['start_idx'] + rb['n_samples']

            assert end >= start, (
                'After accounting for rb_offset there is no sample left in the'
                ' replay buffer. Try to increase the replay buffer or decrease'
                ' the offset.')

            n_seq_per_sample = \
                settings['replay_buffer']['max_n_sequences_per_sample']

            sids = rng.choice(np.arange(start, end),
                              size=min(end-start, n_seq_per_sample))

            moves = [(sid, i) for sid in sids for i in range(len(rb[sid][0]))]

            n_moves = len(moves)
            n_indices = min(n_moves, bs)

            indices = rng.choice(moves, size=n_indices, replace=False)

            sids = np.array([s[0] for s in indices])
            mids = np.array([s[1] for s in indices])
            used_sample_ids = np.unique(sids)

            # process samples
            start = end = 0

            # could evaluate target_fn in parallel (this is slower though!),
            # settings['workings']['rng'] can't be pickled, I only need
            # batch_size anyway
            """
            s = {'training': {'batch_size': settings['training']['batch_size']}}
            ar_list = list()

            for sid in used_sample_ids:
                ar_list.append(
                    sampling_pool.apply_async(target_fn, (rb[sid],), dict(
                        settings=s, val_fn=value_fn,
                        params=params, state=state)))
            """

            for aid, sid in enumerate(used_sample_ids):  # Todo: do I need aid?
                sample = rb[sid]
                move_ids = mids[sids == sid]
                end = start + len(move_ids)

                inputs[start:end] = sample[0][move_ids]
                options[start:end] = sample[1][move_ids]
                idx[start:end] = sample[2][move_ids]

                # """
                target = np.asarray(
                    target_fn(sample, settings=settings, val_fn=value_fn,
                              params=params, state=state))

                targets[start:end] = target[move_ids]

                """
                target = ar_list[aid].get()
                targets[start:end] = np.asarray(target)[move_ids]
                """
                start = end

            check_for_nan(inputs, 'inputs')
            check_for_nan(options, 'options')
            check_for_nan(idx, 'idx')
            check_for_nan(target, 'target')

        except KeyError:
            count += 1
            print(f'\nReplay buffer sampling collision {count}')
            continue

        if settings['output'].get('save_sample_use', False) is True:
            update_sample_use_count(
                settings, rb, used_sample_ids, sids, mids)

        # Todo: could add additional shuffling
        return inputs, options, idx, targets


# If the rb is full then I can now remove the last (new) samples, instead of
# the oldest ones (to test on vs off policy training), in that case this count
# is meaningless (will just be all 1 because the other samples never make
# it this far)
def update_sample_use_count_sequential(settings, used_sample_ids):
    if 'sample_use' not in settings['workings']:
        settings['workings']['sample_use'] = dict()
        settings['workings']['sample_use']['last_idx'] = -1
        settings['workings']['sample_use']['d'] = list()

    for sid in used_sample_ids:
        assert sid > settings['workings']['sample_use']['last_idx']
        for i in range(settings['workings']['sample_use']['last_idx'], sid-1):
            settings['workings']['sample_use']['d'].append('0\n')
        settings['workings']['sample_use']['d'].append('1\n')
        settings['workings']['sample_use']['last_idx'] = sid

    if len(settings['workings']['sample_use']['d']) > 500:
        save_path = settings['workings']['save_path']
        save_path = os.path.join(save_path, 'sample_use.txt')

        with open(save_path, 'a') as f:
            for el in settings['workings']['sample_use']['d']:
                f.write(el)

        settings['workings']['sample_use']['d'] = list()


# Do one game at a time
def sample_from_rb_sequential(rb, settings, params, state):  # called in main process
    # print('SAMPLE FROM RB SEQUENTIAL')
    """ Draw random samples from the replay buffer. """
    bs = settings['training']['batch_size']
    rng = settings['workings']['rng']
    count = 0

    # not using all indices to better deal with concurrent deletions
    rb_offset = settings['replay_buffer']['replay_buffer_sampling_offset']

    if settings['replay_buffer']['only_one_game_in_replay_buffer'] is True:
        rb_offset = 0

    if settings['replay_buffer']['remove_last'] is True:
        rb_offset = 0

    value_fn = get_value_fn(settings)
    target_fn = training.target_functions_sampling.get_target_fn(settings)

    # set up arrays to hold samples (-1 will never occur in game, used in loss
    # functions to identify padding)
    mid_idx = int(rb['start_idx'] + rb['n_samples'] // 2)
    s = rb[mid_idx]

    inputs = -1 * np.ones((bs,) + s[0].shape[1:])
    options = -1 * np.ones((bs,) + s[1].shape[1:], dtype=int)
    idx = -1 * np.ones((bs,), dtype=int)
    targets = -1 * np.ones((bs,))

    # we only want to use each sample once --> save index of first unused sample
    cur_idx = rb.get('next_idx', rb['start_idx'])
    n_seq = settings['replay_buffer']['n_sequences_per_update']

    # size of rb may change while we sample, if we couldn't get a sample
    # --> try again
    while True:
        # with contextlib.suppress(KeyError):
        try:
            # rb_target_size is given in number of moves (if it was given in
            # number of samples then we couldn't ensure a maximum memory usage
            # b/c different samples have different numbers of moves); so
            # rb_offset is also given in number of moves (it would be confusing
            # for the user if it was given in number of samples/games); here we
            # increase the start sample index until the desired number of moves
            # is reached
            start, c = rb['start_idx'], 0
            end = start + rb['n_samples']
            # print('start/end:', start, end)

            while c < rb_offset:
                c += len(rb[start][0])
                start += 1

            # print('start/end:', start, end)
            assert end >= start, (
                'After accounting for rb_offset there is no sample left in the'
                ' replay buffer. Try to increase the replay buffer or decrease'
                ' the offset.')

            # if samples were added more quickly than they are sampled, then
            # the sample after the last used sample may not be in the
            # replay buffer anymore
            cur_idx = max(cur_idx, start)
            # print(f'\n\ncur_idx: {cur_idx}, start_idx={start}, end={end} \n\n')

            # if samples are added more slowly than they are sampled, then we
            # may have to wait until new samples are available
            if cur_idx + n_seq >= end:
                time.sleep(1e-3)
                continue

            start, end = cur_idx, cur_idx + n_seq
            # print('start/end:', start, end)

            # we now have a valid range of sample indices
            moves = [(sid, i)
                     for sid in range(start, end)
                     for i in range(len(rb[sid][0]))]

            n_moves = len(moves)
            n_indices = min(n_moves, bs)

            # indices = rng.choice(moves, size=n_indices, replace=False)
            indices = moves[:n_indices]
            # print(' indices:', indices)

            sids = np.array([s[0] for s in indices])
            mids = np.array([s[1] for s in indices])
            used_sample_ids = np.unique(sids)

            # process samples
            start = end = 0

            # Todo: could evaluate target_fn in parallel (not sure if the
            #       overhead is worth it)
            for sid in used_sample_ids:
                sample = rb[sid]
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

            rb['next_idx'] = cur_idx + n_seq

            check_for_nan(inputs, 'inputs')
            check_for_nan(options, 'options')
            check_for_nan(idx, 'idx')
            check_for_nan(target, 'target')

        except KeyError:
            count += 1
            print(f'\nReplay buffer sampling collision {count}')
            continue

        if settings['output'].get('save_sample_use', False) is True:
            update_sample_use_count_sequential(
                settings, used_sample_ids)

        """
        print(' inputs:', inputs)
        print(' options:', options)
        print(' idx:', idx)
        print(' targets:', targets)
        raise RuntimeError('...')
        """

        # Todo: could add additional shuffling
        return inputs, options, idx, targets


def sample_from_rb(rb, settings, params, state):  # called in main process
    if settings['replay_buffer']['sequential_sampling'] is True:
        return sample_from_rb_sequential(rb, settings, params, state)

    return sample_from_rb_mixed(rb, settings, params, state)

