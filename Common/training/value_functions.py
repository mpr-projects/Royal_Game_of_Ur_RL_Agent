# Functions returning the value estimate of a state, used by target functions;
# these functions take individual samples, no batches!
import jax
import jax.numpy as jnp
import numpy as np


"""
def get_Q_vals_single(board, apply, params, state, bs):
    # single evaluation of forward function, bs must be larger/equal n_moves
    n_moves = len(board)

    board = jnp.pad(
        board,
        ((0, bs-n_moves), (0, 0), (0, 0), (0, 0)),
        constant_values=0)

    (Q, _), _ = apply(params, state, board)
    return np.asarray(Q[:n_moves])


# Todo: this would work for single as well!
def get_Q_vals_multiple(board, apply, params, state, bs, n_calls):
    # longer game, multiple evaluations of forward function are required
    n_moves = len(board)
    Q = list()

    for i in range(n_calls-1):
        b = board[i*bs:(i+1)*bs]
        (q, _), _ = apply(params, state, b)
        Q.append(np.asarray(q))

    b = np.zeros((bs,) + board.shape[1:])
    n_remaining = n_moves - (n_calls-1)*bs
    b[:n_remaining] = board[-n_remaining:]
    (q, _), _ = apply(params, state, b)

    Q.append(np.asarray(q[:n_remaining]))
    return np.concatenate(Q)


def get_Q_vals(board, settings, apply, params, state):
    n_moves = len(board)
    bs = settings['training']['batch_size']
    n_calls = int(np.ceil(n_moves / bs))

    if n_calls == 1:
        return get_Q_vals_single(board, apply, params, state, bs)

    return get_Q_vals_multiple(board, apply, params, state, bs, n_calls)
# """


def get_Q_vals(board, settings, params, state):
    n_moves = len(board)
    bs = settings['training']['batch_size']
    n_calls = int(np.ceil(n_moves / bs))
    Q = list()

    for i in range(n_calls-1):
        b = board[i*bs:(i+1)*bs]
        (q, _), _ = apply(params, state, b)
        Q.append(np.asarray(q))

    b = np.zeros((bs,) + board.shape[1:])
    n_remaining = n_moves - (n_calls-1)*bs
    b[:n_remaining] = board[-n_remaining:]
    (q, _), _ = apply(params, state, b)

    Q.append(np.asarray(q[:n_remaining]))
    return np.concatenate(Q)


def value_fn_Q(params, state, sample, settings, *, single_board):
    board, _, idx, _ = sample

    if single_board is True:  # add 'sample' dimension
        board = board[None, ...]

    Q = get_Q_vals(board, settings, params, state)

    if single_board is True:
        return Q[0, idx]

    return np.take_along_axis(Q, idx[:, None], axis=-1)[..., 0]


def value_fn_Q_mc_off_policy(params, state, sample, settings, *, single_board):
    board, options, idx, _ = sample

    if single_board is True:
        board = board[None, ...]

    Q = get_Q_vals(board, settings, params, state)

    mask = (options == -1)
    Q = np.take_along_axis(Q, options, axis=-1)
    Q = np.where(mask, -float('inf'), Q)
    Q = np.exp(Q)
    Q = Q / Q.sum(axis=-1, keepdims=True)

    selected = (options == idx[:, None])
    return Q[selected]


def value_fn_Q_off_policy(params, state, sample, settings, *, single_board):
    assert single_board is False, \
        'value_fn_Q_off_policy is not implemented for single boards.'

    board, options, _, _ = sample
    Q = get_Q_vals(board, settings, params, state)

    mask = (options == -1)
    Q_max = np.take_along_axis(Q, options, axis=-1)
    Q_max = np.where(mask, -float('inf'), Q_max)
    Q_max = np.max(Q_max, axis=-1, keepdims=True)[:, 0]

    return Q_max


def get_ac_vals(board, settings, params, state):
    n_moves = len(board)
    bs = settings['training']['batch_size']
    n_calls = int(np.ceil(n_moves / bs))
    val = list()

    for i in range(n_calls-1):
        b = board[i*bs:(i+1)*bs]
        (_, v), _ = apply(params, state, b)
        val.append(np.asarray(v))

    b = np.zeros((bs,) + board.shape[1:])
    n_remaining = n_moves - (n_calls-1)*bs
    b[:n_remaining] = board[-n_remaining:]
    (_, v), _ = apply(params, state, b)

    val.append(np.asarray(v[:n_remaining]))
    return np.concatenate(val)


def value_fn_ac(params, state, sample, settings, *, single_board):
    assert single_board is False, \
        'value_fn_ac is not implemented for single boards.'

    board, _, _, _ = sample
    val = get_ac_vals(board, settings, params, state)
    return val[:, 0]


def value_fn_dummy(*args):
    raise RuntimeError('This function should never be called.')



def get_value_fn(settings):
    method = settings['training']['method']

    if method in ['q_td0', 'q_tdn', 'q_tdl']:
        return value_fn_Q 

    if method in ['q_mc_off_policy']:
        return value_fn_Q_mc_off_policy

    if method in ['q_td0_off_policy', 'q_tdn_off_policy', 'q_tdl_off_policy']:
        return value_fn_Q_off_policy

    if method in ['ac_td0', 'ac_tdn', 'ac_tdl', 'ac_mc_off_policy']:
        return value_fn_ac

    if method in ['q_mc', 'policy_mc', 'ac_mc']:  # value function is not used by mc, return dummy
        return value_fn_dummy

    raise NotImplementedError(f'Method {method} is not implemented.')
