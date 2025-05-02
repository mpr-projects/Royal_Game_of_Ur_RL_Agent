"""
These functions will be applied to the sampled data to obtain the 'final'
target to use in training.

These function take an entire game as input.
"""
import jax.numpy as jnp
import numpy as np



# target already contains correct data
def target_fn_passthrough(sample, *args, **kwargs):
    return sample[3]


def target_fn_mc_off_policy(sample, *, settings, val_fn, params, state, p=False):
    # using importance sampling, target = sample[3] contains gamma * G_t / mu_t,
    # now I ned to multiply by pi_t

    # sample[3]: [G_T / mu_T, gamma / mu_(T-1), gamma / mu_(T-2), ...]
    pi = val_fn(params, state, sample, settings, single_board=False)
    rv = np.cumprod(pi * sample[3])

    if p:
        print('-- rv:', rv)

    rv_min = settings.get('mc_off_policy_min_factor', -1)  # min/max rewards are assumed to be -/+1
    rv_max = settings.get('mc_off_policy_max_factor', +1)

    rv = np.maximum(rv_min, np.minimum(rv_max, rv))
    if p:
        print('-- rv post limits:', rv)

    return rv


def target_fn_td0(sample, *, settings, val_fn, params, state):
    gamma = settings['training'].get('gamma', 0.999)

    # get value of next move
    val = gamma * val_fn(params, state, sample, settings, single_board=False)
    val[1:] = val[:-1]  # note, first entry is last move, last entry is first move,  Todo: double check this (and improve description)

    # sample[3] contains [R, 0, 0, 0, ...]
    val[0] = sample[3][0]

    return val


def target_fn_tdn(sample, *, settings, val_fn, params, state):  # n-step td-learning
    gamma = settings['training'].get('gamma', 0.999)
    td_n = settings['training']['td_n'] + 1

    # get value of next moves
    val = val_fn(params, state, sample, settings, single_board=False)

    val[td_n:] = val[:-td_n]  # note, first entry is last move, last entry is first move,  Todo: same as above
    val[td_n:] *= gamma**td_n

    # sample[3] contains [R, 0, 0, 0, ...]
    val[:td_n] = sample[3][0]
    val[:td_n] *= gamma**np.arange(td_n)

    return val


def target_fn_tdl(sample, *, settings, val_fn, params, state):
    gamma = settings['training'].get('gamma', 0.999)
    alpha = settings['training'].get('alpha', 0.9)

    # get value of next move
    val = val_fn(params, state, sample, settings, single_board=False)
    target = np.zeros_like(val)
    target[0] = sample[3][0]

    # Todo: is this loop expensive? Is there a better way?
    for i in range(1, len(val)):
        target[i] = gamma * (alpha*target[i-1] + (1-alpha)*val[i-1])

    return target


def get_target_fn(settings):
    method = settings['training']['method']

    if method in ['q_mc', 'policy_mc', 'ac_mc']:
        return target_fn_passthrough

    if method in ['q_mc_off_policy', 'ac_mc_off_policy']:
        return target_fn_mc_off_policy

    if method in ['q_td0', 'q_td0_off_policy', 'ac_td0']:
        return target_fn_td0

    if method in ['q_tdn', 'q_tdn_off_policy', 'ac_tdn']:
        return target_fn_tdn

    if method in ['q_tdl', 'q_tdl_off_policy', 'ac_tdl']:
        return target_fn_tdl

    raise NotImplementedError(f'Method {method} is not implemented.')
