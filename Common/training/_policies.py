import numpy as np
import jax.numpy as jnp
import argparse


def random_policy(options, rng, **kwargs):
    # return rng.choice(np.array(options), axis=-1)
    n_games = len(options)
    idx = np.zeros(n_games, dtype=int)

    for i in range(n_games):
        o = options[i]
        o = o[o != -1]

        if len(o) != 0:
            idx[i] = rng.choice(o)

    return idx


def get_random_policy(**kwargs):
    return random_policy


def prob_matching_policy(probabilities, rng, **kwargs):  # Todo: probably not up to date
    probs = np.asarray(probabilities)
    probs = probs - np.amax(probs)
    probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return rng.choice(np.arange(16), p=probs, axis=-1)


def get_prob_matching_policy(**kwargs):  # Todo: probably not up to date
    return prob_matching_policy


def greedy_policy(probabilities, **kwargs):
    return jnp.argmax(probabilities, axis=-1)


def get_greedy_policy(**kwargs):
    return greedy_policy


def get_eps_greedy_policy(eps, ref_policy, **kwargs):
    def policy(probabilities, rng, **kwargs):
        if rng.random() < eps:
            return ref_policy(probabilities=probabilities, rng=rng, **kwargs)
        return jnp.argmax(probabilities, axis=-1)
    return policy


def get_step_policy(epochs, policies, settings, **kwargs):
    assert len(epochs) + 1 == len(policies)
    epoch = settings['workings']['epoch']

    pid = np.logical_and(
        [True] + list(epoch > np.array(epochs)),
        list(epoch <= np.array(epochs)) + [True]
    )
    idx = np.where(pid == True)[0][0]
    return policies[idx](settings=settings, **kwargs)


def get_linearly_decaying_eps_greedy_policy(max_eps, min_eps, ref_policy,
                                            settings, **kwargs):
    epoch = settings['workings']['epoch']
    max_epochs = settings['training']['max_epochs']
    eps = max_eps - (max_eps - min_eps) / max_epochs * epoch
    return get_eps_greedy_policy(eps, ref_policy)


def get_power_decaying_eps_greedy_policy(max_eps, min_eps, p, ref_policy,
                                         settings, **kwargs):
    # p > 1 means eps decays slowly at first and then quickly,
    # p == 1 is equivalent to linear decay
    # p < 1 means eps decays quickly at first and then more slowly
    epoch = settings['workings']['epoch']
    max_epochs = settings['training']['max_epochs']
    eps = min_eps + (max_eps - min_eps) * (1 - (epoch / max_epochs)**p)
    return get_eps_greedy_policy(eps, ref_policy)


def get_policy(idx, settings):
    # training_policies have already been imported in sample_creation.
    from training_policies import policies
    policy_fn, kwargs = policies[idx % len(policies)]
    return policy_fn(settings=settings, **kwargs)


# -----------------------------------------------------------------------------
#  Helper to visualize power decay
# -----------------------------------------------------------------------------
def _helper_visualize_power_decay(max_eps, min_eps, p, ax=None):
    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, num=100)
    eps = min_eps + (max_eps - min_eps) * (1 - x**p)

    if ax is None:
        plt.plot(x, eps, label=f'p={p:.2f}')
        plt.xlabel('Training Progress')
        plt.ylabel('eps')
        plt.show()

    else:
        ax.plot(x, eps, label=f'p={p:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='VisualizePowerDecayPolicy')
    parser.add_argument('max_eps', type=float)
    parser.add_argument('min_eps', type=float)
    parser.add_argument('p_vals', nargs='+', type=float)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for p in args.p_vals:
        _helper_visualize_power_decay(args.max_eps, args.min_eps, p, ax=ax)

    plt.xlabel('Training Progress')
    plt.ylabel('eps')
    plt.legend()
    plt.show()
