"""
Input is a vector with 14 elements giving the number of stones in the pits (5)
and in the store (1) of each player (x2). Output is a vector with 5 components,
one for each possible pit player 0 can choose (the machine player always plays
from the view of player 0 -- for player 1 the board is flipped).

Todo: normalize inputs (divide by 4 or -4 / 4)
"""
import jax
import jax.numpy as jnp
import haiku as hk
import functools
import numpy as np

import training.CustomHaikuModules as chk


def forward_Q_(x, return_activations=True, p=False):

    if p is True:
        print('\n\nforward_Q, input:', x, '\n\n')

    # normalize input
    x = (x - 4) / 4

    if p is True:
        print('normalized:', x, '\n', x.shape)

    n_ops = 4
    # print('forward_Q:')
    # print(' x:', x.shape)

    # when structuring the model it helps to see how activations behave
    if return_activations is True:
        activations = jnp.zeros((n_ops+2, 2))
        activations = activations.at[0, 0].set(jnp.mean(x))
        activations = activations.at[0, 1].set(jnp.var(x))

    for i in range(n_ops):
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)

        if return_activations is True:
            activations = activations.at[i+1, 0].set(jnp.mean(x))
            activations = activations.at[i+1, 1].set(jnp.var(x))

        if p is True:
            print('i', i, ':', x.shape)


    x = hk.Linear(6)(x)
    x = jax.nn.tanh(x)

    if p is True:
        print('final', ':', x, '\n', x.shape)
        raise RuntimeError('...')

    if return_activations is True:
        activations = activations.at[n_ops+1, 0].set(jnp.mean(x))
        activations = activations.at[n_ops+1, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x, jnp.ones(1)


def reshape_board_(board):
    x = jnp.zeros((len(board), 2, 7))
    x = x.at[:, 0].set(board[:, :7])
    x = x.at[:, 1, :6].set(board[:, 12:6:-1])
    x = x.at[:, 1, 6].set(board[:, -1])
    return x


def reshape_board(board):  # Todo: could potentially reshape in place
    x = jnp.zeros((len(board), 2, 7))
    x = x.at[:, 0].set(board[:, :7])
    x = x.at[:, 1].set(board[:, 13:6:-1])
    return x


def forward_Q(x, return_activations=True, **kwargs):
    # reshape into board form
    x = reshape_board(x)

    ocs = 128
    # define operations
    ops = (
        hk.LayerNorm(axis=[1, 2], create_scale=False, create_offset=False),
        chk.Conv2DSW(output_channels=ocs, kernel_shape=2, gain='relu'),
        'save_skip',
    )

    for i in range(3):
        if i != 0:
            ops += (hk.LayerNorm(axis=[1, 2], create_scale=False, create_offset=False),)

        ops += (
            jax.nn.relu,
            chk.Conv2DSW(output_channels=ocs, kernel_shape=2, gain='relu'),
            jax.nn.relu,
            chk.Conv2DSW(output_channels=ocs, kernel_shape=2, gain='relu'),
            'add_skip',
            'save_skip',
        )

    for ocs in [128, 128, 128]:
        ops += (
            f'downsample_skip_{ocs}',
            'normalize_skip',
        )

        for i in range(3):
            ops += (
                hk.LayerNorm(axis=[1, 2], create_scale=False, create_offset=False),
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=2, gain='relu'),  # , stride=(2 if i==0 else 1)
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=2, gain='relu'),
                'add_skip',
                'save_skip',
            )

    ops += (
        hk.LayerNorm(axis=[1, 2], create_scale=False, create_offset=False),
        jax.nn.relu,
        hk.Flatten(),
        chk.LinearSW(6, gain='relu'),  # 16 instead of 24 because we only look at player 0
        jax.nn.tanh,
    )


    # run operations
    if return_activations is True:
        n_ops = len(ops)
        activations = jnp.zeros((n_ops+1, 2))
        n_ = 0

    x_prev = jnp.ones(1)
    # oid = 0 # temp

    for op in ops:
        if return_activations is True:
            activations = activations.at[n_, 0].set(jnp.mean(x))
            activations = activations.at[n_, 1].set(jnp.var(x))
            n_ += 1

        if isinstance(op, str):
            if op == 'save_skip':
                x_prev = x

            elif op == 'add_skip':
                x = x + x_prev

            elif op == 'normalize_skip':
                x_prev = (x_prev - jnp.mean(x_prev, axis=[1, 2], keepdims=True)) \
                    / jnp.std(x, axis=[1, 2], keepdims=True)

            elif op[:15] == 'downsample_skip':
                oc = int(op[16:])
                x_prev = chk.Conv2DSW(output_channels=oc, kernel_shape=1)(x_prev)  # , stride=2

        else:
            x = op(x)

        # all zeros can cause a problem in convolution
        x = jnp.where(x == 0, 1e-7, x)

        # print('op', oid, ':', op, x[:12])
        # oid += 1

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x, jnp.ones(1)


def forward_Policy(x, return_activations=True, p=False):  # p can be used for printing (debugging)
    n_ops = 4

    # when structuring the model it helps to see how activations behave
    if return_activations is True:
        activations = jnp.zeros((n_ops+1, 2))
        activations = activations.at[0, 0].set(jnp.mean(x))
        activations = activations.at[0, 1].set(jnp.var(x))

    for i in range(n_ops):
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)

        if return_activations is True:
            activations = activations.at[i+1, 0].set(jnp.mean(x))
            activations = activations.at[i+1, 1].set(jnp.var(x))

    x = hk.Linear(6)(x)
    # jax.nn.softmax,  # do in loss fn after masking

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x, jnp.ones(1)


def forward_ActorCritic(x, return_activations=True):  # Todo: activations are incorrect at the moment
    # common operations
    n_ops = 2

    # when structuring the model it helps to see how activations behave
    if return_activations is True:
        activations = jnp.zeros((n_ops+1, 2))
        activations = activations.at[0, 0].set(jnp.mean(x))
        activations = activations.at[0, 1].set(jnp.var(x))

    for i in range(n_ops):
        x = hk.Linear(128)(x)
        x = jax.nn.relu(x)

        if return_activations is True:
            activations = activations.at[i+1, 0].set(jnp.mean(x))
            activations = activations.at[i+1, 1].set(jnp.var(x))

    # pi-operations
    n_ops = 2
    x_pi = jnp.copy(x)

    for i in range(n_ops):
        x_pi = hk.Linear(128)(x_pi)
        x_pi = jax.nn.relu(x_pi)

    x_pi = hk.Linear(6)(x_pi)
    # jax.nn.softmax,  # do in loss fn after masking

    # val-operations
    n_ops = 2
    x_val = x

    for i in range(n_ops):
        x_val = hk.Linear(128)(x_val)
        x_val = jax.nn.relu(x_val)

    x_val = hk.Linear(1)(x_val)
    x_val = jax.nn.tanh(x_val)

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x_pi, x_val


def get_forward_fn(settings):
    method = settings['training']['method']

    if method in ['q_mc', 'q_td0', 'q_tdn', 'q_tdl', 'q_mc_off_policy', 'q_td0_off_policy', 'q_tdn_off_policy', 'q_tdl_off_policy']:
        return forward_Q

    if method in ['policy_mc']:
        return forward_Policy

    if method in ['ac_mc', 'ac_mc_off_policy', 'ac_td0', 'ac_tdn', 'ac_tdl']:
        return forward_ActorCritic

    raise NotImplementedError(f'Method {method} is not implemented.')

