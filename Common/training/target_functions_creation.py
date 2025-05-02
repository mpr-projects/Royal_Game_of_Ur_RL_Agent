import numpy as np


# sample: (board, options, idx, R)
def target_fn_mc(sample, aux, *, settings, **kwargs):
    gamma = settings['training'].get('gamma', 1.0)
    aux['val'] = gamma * aux.get('val', sample[3])
    return aux['val'], aux  # first argument is target, second is aux

def target_fn_mc(sample, aux, *, settings, **kwargs):
    gamma = settings['training'].get('gamma', 1.0)
    R, n_moves = sample[3], len(sample[0])
    return R * gamma**np.arange(n_moves), aux


def target_fn_mc_off_policy(sample, aux, *, settings, val_fn, params, state):
    # using importance sampling, save gamma * G_t / mu_t, during sample
    # creation I have to multiply by pi_t (cummulatively, for all t to T)
    # Todo: explain this more clearly
    gamma = settings['training'].get('gamma', 1.0)
    mu = val_fn(params, state, sample, settings, single_board=True)
    rv = gamma * aux.get('val', sample[3]) / mu
    aux['val'] = 1.0
    return rv, aux  # first argument is target, second is aux

def target_fn_mc_off_policy(sample, aux, *, settings, val_fn, params, state):
    # using importance sampling, save gamma * G_t / mu_t, during sample
    # creation I have to multiply by pi_t (cummulatively, for all t to T)
    # Todo: explain this more clearly
    gamma = settings['training'].get('gamma', 1.0)
    mu = val_fn(params, state, sample, settings, single_board=False)

    n_moves = len(sample[0])
    rv = gamma * np.ones(n_moves)
    rv[0] = sample[3]
    rv /= mu

    return rv, aux  # first argument is target, second is aux


def target_fn_td(sample, aux, *, settings, val_fn, params, state):
    # saves [R, 0, 0, ...]
    rv = aux.get('val', sample[3])
    aux['val'] = 0.0
    return rv, aux


def target_fn_td(sample, aux, *, settings, val_fn, params, state):
    # saves [R, 0, 0, ...]
    n_moves = len(sample[0])
    rv = np.zeros(n_moves)
    rv[0] = sample[3]
    return rv, aux




def get_target_fn(settings):
    method = settings['training']['method']

    if method in ['q_mc', 'policy_mc', 'ac_mc']:
        return target_fn_mc

    if method in ['q_mc_off_policy', 'ac_mc_off_policy']:
        return target_fn_mc_off_policy

    if method in ['q_td0', 'q_tdn', 'q_tdl', 'q_td0_off_policy', 'q_tdn_off_policy', 'q_tdl_off_policy', 'ac_td0', 'ac_tdn', 'ac_tdl']:
        return target_fn_td

    raise NotImplementedError(f'Method {method} is not implemented.')
