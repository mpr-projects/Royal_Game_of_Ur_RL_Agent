import jax
import jax.numpy as jnp
import optax
import numpy as np



# -----------------------------------------------------------------------------
#  Updated
# -----------------------------------------------------------------------------
def get_mask_inputs(inputs):
    # we may have variable-length batches, only include valid inputs
    ax = tuple(np.arange(1, inputs.ndim))
    mask = jnp.logical_not((inputs == -1).all(axis=ax))
    mask = jax.lax.stop_gradient(mask)

    # for some reason constant inputs lead to NaN gradients (something wrong in forward?)
    key = jax.random.PRNGKey(43)
    ran = jax.random.normal(key, inputs.shape)
    inputs = jnp.where(jnp.expand_dims(mask, axis=ax), inputs, ran)

    return mask, inputs


def loss_fn_Q(params, state, inputs, idx, target, **kwargs):
    mask, inputs = get_mask_inputs(inputs)
    (Q, _), _ = apply(params, state, inputs)
    rv = jnp.take_along_axis(Q, idx[:, None], axis=-1)[..., 0]

    loss = optax.l2_loss(rv, target)
    loss = jnp.where(mask, loss, 0)
    return loss.sum() / mask.sum()


def loss_fn_actor_critic(params, state, inputs, idx, target, options):
    i_mask, inputs = get_mask_inputs(inputs)
    o_mask = (options == -1)

    (pi, val), _ = apply(params, state, inputs)

    lv = optax.l2_loss(val[:, 0], target)
    lv = jnp.where(i_mask, lv, 0)

    pi_max = jnp.take_along_axis(pi, options, axis=-1)
    pi_max = jnp.where(o_mask, -float('inf'), pi_max)
    pi_max = jnp.max(pi_max, axis=-1, keepdims=True)

    shifted = pi - jax.lax.stop_gradient(pi_max)  # shifting means all x <= 0, ie no buffer overflow

    shifted_logsumexp = jnp.take_along_axis(shifted, options, axis=-1)
    shifted_logsumexp = jnp.where(o_mask, -float('inf'), shifted_logsumexp)
    shifted_logsumexp = jnp.exp(shifted_logsumexp)
    shifted_logsumexp = jnp.sum(shifted_logsumexp, axis=-1)
    shifted_logsumexp = jnp.log(shifted_logsumexp)

    log_softmax = jnp.take_along_axis(shifted, idx[:, None], axis=-1)
    log_softmax = log_softmax[:, 0] - shifted_logsumexp

    lp = (jax.lax.stop_gradient(val[:, 0]) - target) * log_softmax
    lp = jnp.where(i_mask, lp, 0)

    return (lv + lp).sum() / i_mask.sum()


def loss_fn_REINFORCE(params, state, inputs, idx, target, options):
    i_mask, inputs = get_mask_inputs(inputs)
    o_mask = (options == -1)

    (pi, _), _ = apply(params, state, inputs)

    pi_max = jnp.take_along_axis(pi, options, axis=-1)
    pi_max = jnp.where(o_mask, -float('inf'), pi_max)
    pi_max = jnp.max(pi_max, axis=-1, keepdims=True)

    shifted = pi - jax.lax.stop_gradient(pi_max)  # shifting means all x <= 0, ie no buffer overflow

    shifted_logsumexp = jnp.take_along_axis(shifted, options, axis=-1)
    shifted_logsumexp = jnp.where(o_mask, -float('inf'), shifted_logsumexp)
    shifted_logsumexp = jnp.exp(shifted_logsumexp)
    shifted_logsumexp = jnp.sum(shifted_logsumexp, axis=-1)
    shifted_logsumexp = jnp.log(shifted_logsumexp)

    log_softmax = jnp.take_along_axis(shifted, idx[:, None], axis=-1)
    log_softmax = log_softmax[:, 0] - shifted_logsumexp

    target = jnp.where(i_mask, target, 0)
    loss = - target * log_softmax
    return loss.sum() / i_mask.sum()



def get_loss_fn(settings):
    method = settings['training']['method']
    print('get_loss_fn', method)

    m = method.split('_')[0]  # must be in ['q', 'ac', 'policy']

    if m == 'q':
        return loss_fn_Q

    elif m == 'ac':
        return loss_fn_actor_critic

    elif method == 'policy_mc':
        return loss_fn_REINFORCE

    else:
        raise NotImplementedError(f'Method {method} is not implemented.')








# -----------------------------------------------------------------------------
#  Old 
# -----------------------------------------------------------------------------

def loss_fn_REINFORCE__(params, state, inputs, idx, R, options):
    (pi, _), _ = apply(params, state, inputs)
    pi = pi[0]
    pi_max = jnp.max(pi[options], axis=-1, keepdims=True)
    shifted = pi - jax.lax.stop_gradient(pi_max)  # shifting means all x <= 0, ie no buffer overflow
    shifted_logsumexp = jnp.log(
        jnp.sum(jnp.exp(shifted[options]), axis=-1, keepdims=True))
    log_softmax = shifted[idx] - shifted_logsumexp
    return -(R * log_softmax).mean(), 0


def loss_fn_actor_critic_mc_off_policy(params, state, inputs, idx, R, options):
    (pi, val), _ = apply(params, state, inputs)
    pi, val = pi[0], val[0]

    pi_max = jnp.max(pi[options], axis=-1, keepdims=True)
    shifted = pi - jax.lax.stop_gradient(pi_max)  # shifting means all x <= 0, ie no buffer overflow

    # importance sampling, account for difference in behavioural and optimal policy
    R = jax.lax.stop_gradient(R / jnp.exp(shifted[idx]))
    R = jnp.maximum(jnp.minimum(R, 10), -10)

    shifted_logsumexp = jnp.log(
        jnp.sum(jnp.exp(shifted[options]), axis=-1, keepdims=True))
    log_softmax = shifted[idx] - shifted_logsumexp
    lp = -(R - jax.lax.stop_gradient(val[0])) * log_softmax[0]
    
    lv = optax.l2_loss(val[0], R).mean()

    return lv + lp, R


# not yet used, wrote it in preparation
def loss_fn_double_Q(params, state, inputs, idx, R, **kwargs):
    (Q1, Q2, _), _ = apply(params, state, inputs)
    rv = min(Q1[0, idx], Q2[0, idx])
    return optax.l2_loss(rv, R).mean(), rv


"""
def loss_fn_Q_off_policy(params, state, inputs, idx, target, options):   # <------- updated!  check if I'm doing the right thing here
    (Q, _), _ = apply(params, state, inputs)

    rv = jnp.take_along_axis(
        Q, jax.lax.stop_gradient(options), axis=-1)

    rv = jnp.where(options == -1, -float('inf'), rv)
    rv = jnp.max(rv, axis=-1)
    return optax.l2_loss(rv, target).mean()


loss_fn_Q_off_policy = loss_fn_Q
"""


# not yet used, wrote it in preparation
def loss_fn_double_Q_off_policy(params, state, inputs, idx, R, options):
    (Q1, Q2, _), _ = apply(params, state, inputs)
    rv = jnp.max(jnp.minimum(Q1[0, options], Q2[0, options]))
    return optax.l2_loss(rv, R).mean(), rv


"""
def get_loss_fn(method):
    m = method.split('_')[0]  # must be in ['q', 'ac', 'policy']

    if m == 'q':
        return loss_fn_Q

    elif m == 'ac':
        return loss_fn_actor_critic

    elif m == 'policy':
        return loss_fn_REINFORCE

    else:
        raise NotImplementedError(f'Method {method} is not implemented.')

    # old
    if method in ['q_mc', 'q_td0', 'q_tdl']:
        return loss_fn_Q

    if method in ['q_td0_off_policy', 'q_tdl_off_policy']:
        return loss_fn_Q_off_policy

    if method in ['policy_mc', 'policy_td0', 'policy_tdl']:
        return loss_fn_REINFORCE

    if method in ['ac_mc', 'ac_td0', 'ac_tdl']:
        return loss_fn_actor_critic

    if method in ['ac_mc_off_policy']:
        return loss_fn_actor_critic_mc_off_policy

    raise NotImplementedError(f'Method {method} is not implemented.')
"""
