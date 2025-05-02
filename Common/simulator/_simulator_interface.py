""" --- General (not problem-specific) File ---
The function 'get_machine_fn_rl' creates the interface between the RL-framework
and the simulator. It loads the model's parameters and create a machine function
from it (mostly used for visual testing).
"""
import os
import jax
import training._utils
import interface.machine_agent

import training._policies


def get_machine_fn_rl(data_folder, epoch=-1, backend='gpu', print_probs=True, policy='greedy', is_human_game=False):
    # datafolder can be i) a folder containing settings.yaml and checkpoints
    #  of a model or ii) a folder containing multiple folders like i). In the
    #  latter case the latest folder will be chosen.
    jax.config.update('jax_platform_name', backend)
    jax.default_device = jax.devices(backend)[0]

    # check if data_folder contains 'settings.yaml', if not load the
    # latest subfolder
    sf = os.path.join(data_folder, 'settings.yaml')

    if not os.path.isfile(sf):
        folders_list = os.listdir(data_folder)
        folders_list.sort()
        data_folder = os.path.join(data_folder, folders_list[-1])
        sf = os.path.join(data_folder, 'settings.yaml')

        assert os.path.isfile(sf), (
            f'Could not find "settings.yaml" in the provided folder or its'
            f' latest subfolder ({data_folder}).')

    # load data
    settings = training._utils.load_settings_from_file(sf)

    backend = settings['output']['evaluation_backend']
    _, apply = training._utils.get_init_apply(settings, backend)

    params, state = training._utils.load_existing_params(
        data_folder, epoch, tree_names=['params', 'state'])

    if policy == 'greedy':
        policy = training._policies.greedy_policy

    return interface.machine_agent.create(
        params, state, policy, apply, print_probs=print_probs, is_human_game=is_human_game)
