"""  --- Problem-specific File ---   Todo: is it really problem_specific?
"""
import numpy as np
from simulator.machine_game import flip_board, flip_indices
from training._policies import random_policy, get_policy


def create(params, state, policy, apply, print_probs=False, **kwargs):
    """
    Interface between model and game. Plays according to given policy.

    If parameter 'policy' is an integer then it refers to the desired policy
    in file settings/training_policies.py. Otherwise a policy can be passed
    directly.
    """
    if isinstance(policy, int):
        policy = get_policy(policy)

    def machine_player(inputs, options, current_player, rng):
        # print('Machine Player:')
        # print(' shapes:', inputs.shape, options.shape, current_player.shape)
        # print(' options:', options)
        pi = None

        # flip board for player 1 s.t. everything is from view of player 1
        mask_player = (current_player == 1)

        ax = tuple(range(1, inputs.ndim))
        inputs = np.where(np.expand_dims(mask_player, axis=ax),
                          flip_board(inputs), inputs)

        # flip_indices takes care of -1 (which is used as padding)
        options = np.where(mask_player[:, None], flip_indices(options), options)

        if policy is not random_policy:
            (pi, val), _ = apply(params, state, inputs)
            pi = np.array(pi)  # maybe asarray and allow editing?
            val = np.array(val)

            # print('shapes:', pi.shape, options.shape)
            # print('pi:', pi)
            n_outputs = pi.shape[1]  # 5
            masks = np.array(
                [np.isin(range(n_outputs), o, assume_unique=True, invert=True)
                 for o in options])
            # print('masks:', masks)
            pi[masks] = -float('inf')
            # print('pi:', pi)

            if print_probs is True:
                print('pi:', pi, val, options)

            """
            pi, val = pi[0], val[0]
            mask = np.array([i for i in range(len(pi)) if i not in options])

            if print_probs is True:
                print('pi:', pi, val, options)
                print('   pi available:', pi[np.array(options)])

            pi = pi.at[mask].set(-float('inf'))
            """

        idx = policy(probabilities=pi, options=options, rng=rng)
        # print(' idx:', idx, '(options =', options, ')', pi)
        idx = np.where(mask_player, flip_indices(idx), idx)
        return idx

    return machine_player
