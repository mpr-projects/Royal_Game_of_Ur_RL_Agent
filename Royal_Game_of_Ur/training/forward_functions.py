import jax
import jax.numpy as jnp
import haiku as hk
import functools
import numpy as np

import training.CustomHaikuModules as chk


def forward_Q(x, return_activations=True, p=False):  # p can be used for printing (debugging)
    ocs = 128
    # define operations
    ops = (
        hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
        chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
        'save_skip',
    )

    for i in range(3):
        if i != 0:
            ops += (hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),)

        ops += (
            jax.nn.relu,
            chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
            jax.nn.relu,
            chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
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
                hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),  # , stride=(2 if i==0 else 1)
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
                'add_skip',
                'save_skip',
            )

    ops += (
        hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
        jax.nn.relu,
        hk.Flatten(),
        chk.LinearSW(16, gain='relu'),  # 16 instead of 24 because we only look at player 0
        jax.nn.tanh,
    )


    # run operations
    if return_activations is True:
        n_ops = len(ops)
        activations = jnp.zeros((n_ops+1, 2))
        n_ = 0

    x_prev = jnp.ones(1)

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
                x_prev = (x_prev - jnp.mean(x_prev, axis=[1, 2, 3], keepdims=True)) \
                    / jnp.std(x, axis=[1, 2, 3], keepdims=True)

            elif op[:15] == 'downsample_skip':
                oc = int(op[16:])
                x_prev = chk.Conv2DSW(output_channels=oc, kernel_shape=1)(x_prev)  # , stride=2

        else:
            x = op(x)

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x, jnp.ones(1)


def forward_Policy(x, return_activations=True, p=False):  # p can be used for printing (debugging)
    # define operations
    ops = (
        hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
        chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
        'save_skip',
    )

    for i in range(2):
        if i != 0:
            ops += (hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),)

        ops += (
            jax.nn.relu,
            chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
            jax.nn.relu,
            chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
            'add_skip',
            'save_skip',
        )
    
    for ocs in [128, 128, 128]:
        ops += (
            f'downsample_skip_{ocs}',
            'normalize_skip',
        )

        for i in range(2):
            ops += (
                hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),  # , stride=(2 if i==0 else 1)
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
                'add_skip',
                'save_skip',
            )

    ops += (
        hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
        jax.nn.relu,
        hk.Flatten(),
        chk.LinearSW(16, gain='relu'),  # 16 instead of 24 because we only look at player 0
        # jax.nn.softmax,  # do in loss fn after masking
    )


    # run operations
    if return_activations is True:
        n_ops = len(ops)
        activations = jnp.zeros((n_ops+1, 2))
        n_ = 0

    x_prev = jnp.ones(1)

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
                x_prev = (x_prev - jnp.mean(x_prev, axis=[1, 2, 3], keepdims=True)) \
                    / jnp.std(x, axis=[1, 2, 3], keepdims=True)

            elif op[:15] == 'downsample_skip':
                oc = int(op[16:])
                x_prev = chk.Conv2DSW(output_channels=oc, kernel_shape=1)(x_prev)  # , stride=2

        else:
            x = op(x)

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))
        hk.set_state("activations", activations)

    # returning jnp.ones(1) for compatibility with actor-value output
    return x, jnp.ones(1)



def forward_ActorCritic(x, return_activations=True, p=False):  # p can be used for printing (debugging)
    # define operations
    ops = (
        hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
        chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
        'save_skip',
    )

    for i in range(2):
        if i != 0:
            ops += (hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),)

        ops += (
            jax.nn.relu,
            chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
            jax.nn.relu,
            chk.Conv2DSW(output_channels=64, kernel_shape=3, gain='relu'),
            'add_skip',
            'save_skip',
        )
    
    for ocs in [128, 128, 128]:
        ops += (
            f'downsample_skip_{ocs}',
            'normalize_skip',
        )

        for i in range(2):
            ops += (
                hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),  # , stride=(2 if i==0 else 1)
                jax.nn.relu,
                chk.Conv2DSW(output_channels=ocs, kernel_shape=3, gain='relu'),
                'add_skip',
                'save_skip',
            )

    ops += (hk.LayerNorm(axis=[1, 2, 3], create_scale=False, create_offset=False),)

    ops_pi = (  # policy head
        chk.Conv2DSW(output_channels=2, kernel_shape=1),
        jax.nn.relu,
        hk.Flatten(),
        chk.LinearSW(16, gain='relu'),  # 16 instead of 24 because we only look at player 0
        # jax.nn.softmax,
    )
    
    ops_val = (  # value head
        chk.Conv2DSW(output_channels=1, kernel_shape=1),
        jax.nn.relu,
        hk.Flatten(),
        chk.LinearSW(256, gain='relu'),
        jax.nn.relu,
        chk.LinearSW(1, gain='relu'),
        jax.nn.tanh,
    )

    # run operations
    if return_activations is True:
        n_ops = len(ops) + len(ops_pi) + len(ops_val)
        activations = jnp.zeros((n_ops+1, 2))
        n_ = 0

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
                x_prev = (x_prev - jnp.mean(x_prev, axis=[1, 2, 3], keepdims=True)) \
                    / jnp.std(x, axis=[1, 2, 3], keepdims=True)

            elif op[:15] == 'downsample_skip':
                oc = int(op[16:])
                x_prev = chk.Conv2DSW(output_channels=oc, kernel_shape=1)(x_prev)  # , stride=2

        else:
            x = op(x)

    if return_activations is True:
        activations = activations.at[n_, 0].set(jnp.mean(x))
        activations = activations.at[n_, 1].set(jnp.var(x))

    x_pi = jnp.copy(x)

    for op in ops_pi:
        x_pi = op(x_pi)

        if return_activations is True:
            activations = activations.at[n_, 0].set(jnp.mean(x_pi))
            activations = activations.at[n_, 1].set(jnp.var(x_pi))
            n_ += 1

    x_val = jnp.copy(x)

    for op in ops_val:
        x_val = op(x_val)

        if return_activations is True:
            activations = activations.at[n_, 0].set(jnp.mean(x_val))
            activations = activations.at[n_, 1].set(jnp.var(x_val))
            n_ += 1

    if return_activations is True:
        hk.set_state("activations", activations)

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

