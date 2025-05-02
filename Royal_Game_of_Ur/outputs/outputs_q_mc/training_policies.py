"""
Define the policies to use in sample creation.
"""
from training._policies import (
    random_policy, get_random_policy,
    greedy_policy, get_greedy_policy,
    prob_matching_policy,
    get_eps_greedy_policy,
    get_prob_matching_policy,
    get_step_policy,
    get_linearly_decaying_eps_greedy_policy,
    get_power_decaying_eps_greedy_policy)


policies = [
    [
        get_power_decaying_eps_greedy_policy,
        dict(max_eps=0.05, min_eps=0.0, p=4, ref_policy=random_policy)],
]
