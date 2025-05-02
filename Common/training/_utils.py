import os
import time
import yaml
import functools
import numpy as np
import pickle
import subprocess
import argparse
import datetime
import shutil

import jax
import haiku as hk

import matplotlib.pyplot as plt

from training.forward_functions import get_forward_fn
import training.utils  # problem-specific file


# -----------------------------------------------------------------------------
#  Load Settings
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='RoyalGameOfUrAI',
        description= \
            'Train a neural network based player of the \'Royal Game of Ur\'.')

    parser.add_argument('--settings_folder', default='settings', help=(
        'The settings folder should contain the following files:'
        ' i) settings.yaml, ii) optimizer_settings.py'))

    parser.add_argument('--continue_training', nargs='+', help=(
        'Continue previous training. Takes two arguments: i) the path to the'
        ' output folder of the previous training run and ii) the epoch from'
        ' which to continue training. If ii) is not given then training'
        ' continues from the last epoch. Argument --settings_folder will be'
        ' ignored.'))

    parser.add_argument('--same_folder', action='store_true', help=(
        'If --continue_training is used then this flag determines if a new'
        ' folder should be created or if data should be saved in the existing'
        ' folder.'))

    parser.add_argument('--readme', nargs='+', help=(
        'Information about the training process, stored in file readme.txt.'
        ' Will be ignored if --continue_training is used.'))

    args = parser.parse_args()

    if args.continue_training is not None:
        assert len(args.continue_training) <= 2

    if args.continue_training and (args.readme is not None):
        print('Warning: --readme gets ignored with --continue_training.')

    return args


def load_settings_from_file(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def load_settings_file(args):
    if args.continue_training is not None:
        args.settings_folder = os.path.join(args.continue_training[0])

    settings = load_settings_from_file(
        os.path.join(args.settings_folder, 'settings.yaml'))

    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    method = settings['training']['method']
    save_path = f'outputs/outputs_{method}/{start_time}'

    settings['workings'] = dict(
        epoch=0, start_time=start_time, save_path=save_path,
        settings_folder=args.settings_folder, readme=args.readme)

    training.utils.check_settings(settings)
    return settings


def copy_settings_to_output_folder(settings):
    if settings['workings'].get('copied_settings', False) is True:
        return

    save_path = settings['workings']['save_path']
    os.makedirs(save_path, exist_ok=True)
    print('Using output folder:', save_path)

    sf = settings['workings']['settings_folder']
    shutil.copy2(os.path.join(sf, 'settings.yaml'), save_path)
    shutil.copy2(os.path.join(sf, 'optimizer_settings.py'), save_path)
    shutil.copy2(os.path.join(sf, 'training_policies.py'), save_path)

    if (readme := settings['workings']['readme']) is not None:
        with open(os.path.join(save_path, 'readme.txt'), 'w') as f:
            f.write(' '.join(readme))

    settings['workings']['copied_settings'] = True


def load_new_params(settings):
    rng_key = jax.random.PRNGKey(settings['training']['seed'])
    x = training.utils.get_dummy_input(settings)
    return init(rng=rng_key, x=x)


def load_existing_params(*args, settings=None,
                         tree_names=['params', 'state', 'opt_state']):
    # args contains either (path, epoch) or (path,)
    path, epoch = args[0], -1

    if len(args) == 2:
        epoch = int(args[1])

    available_epochs = [f.split('_') for f in os.listdir(path)]
    available_epochs = [f[0] for f in available_epochs if f[0].isdigit()]
    available_epochs = np.unique(available_epochs)  # also sorts values

    epoch_str = None

    if epoch == -1:
        epoch_str = available_epochs[-1]

    else:
        for e in available_epochs:
            if int(e) == epoch:
                epoch_str = e
                break

    assert epoch_str is not None, f'Coudn\'t find data for epoch {epoch}.'

    if settings is not None:
        settings['workings']['epoch'] = int(epoch_str)

    fpath = os.path.join(path, epoch_str)
    loaded = restore_dicts(fpath, tree_names)
    return [loaded[tn] for tn in tree_names]
# return loaded['params'], loaded['state'], loaded['opt_state']


# -----------------------------------------------------------------------------
#  Initialization
# -----------------------------------------------------------------------------
def initializer_cpu_only():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"plattform\"'


def get_init_apply(settings, backend, jit_compile=True):
    forward = get_forward_fn(settings)
    forward = hk.transform_with_state(forward)
    forward = hk.without_apply_rng(forward)

    if jit_compile is False:
        init, apply = forward.init, forward.apply

    else:
        init = functools.partial(
            jax.jit, static_argnames=['return_activations', 'p'])(  # p is used for debugging to indicate if something should be printed
                forward.init, backend=backend)

        apply = functools.partial(
            jax.jit, static_argnames=['return_activations', 'p'])(
                forward.apply, backend=backend)

    return init, apply


# -----------------------------------------------------------------------------
#  ...
# -----------------------------------------------------------------------------
def check_for_manual_changes(settings):
    fpaths = ['.', settings['workings']['save_path']]

    for fpath in fpaths:
        fname = os.path.join(fpath, 'STOP')

        if os.path.exists(fname):
            try:
                with open(fname) as f:
                    settings['workings']['STOP'] = True
                os.remove(fname)
            except:
                print("\nCould not read file 'STOP'.")

        fname = os.path.join(fpath, 'CHANGE_MAX_EPOCHS')

        if os.path.exists(fname):
            try:
                with open(fname) as f:
                    v = int(f.read())
                settings['training']['max_epochs'] = v
                os.remove(fname)
            except:
                print("\nCould not read file 'CHANGE_MAX_EPOCHS'.")


def measure_speed_start(settings, epoch):
    # measure how long it takes to run one epoch (use average over a range
    # of epochs, starting after the initial setup/compilation)
    if epoch == settings['output']['measure_speed_start_epoch']:
        settings['workings']['ms_start'] = time.time()

def measure_speed_end(settings, epoch):
    # measure how long it takes to run one epoch (use average over a range
    # of epochs, starting after the initial setup/compilation)
    if epoch == settings['output']['measure_speed_end_epoch']:
        ms = (time.time() - settings['workings']['ms_start']) / (
            settings['output']['measure_speed_end_epoch']
            - settings['output']['measure_speed_start_epoch'])

        save_path = settings['workings']['save_path']
        save_path = os.path.join(save_path, 'average_time_per_epoch.txt')

        with open(save_path, 'w') as f:
            f.write(str(ms))


# -----------------------------------------------------------------------------
#  Evaluation
# -----------------------------------------------------------------------------
def evaluate(settings, save_path):
    n_games = settings['output'].get('n_evaluation_games', -1)

    if n_games < 0:
        return

    p = settings['workings'].get('evaluating', None)

    if p is not None:  # we were evaluating, may have finished by now
        if p.poll() is None:  # still evaluating, don't run this one
            return

    # not evaluating any other params, run these ones
    epoch = settings['workings']['epoch']
    device = settings['output'].get('evaluate_backend', 'cpu')

    n_digits = len(str(settings['training']['max_epochs']))
    fname = f'{epoch:0{max(1, n_digits)}d}'

    settings['workings']['evaluating'] = \
        subprocess.Popen(
            ["python", "evaluate.py",
             '--evaluated_player', save_path,
             '--n_games', str(n_games),
             '--device', device])  # ,
             # '--fname', f'epoch_{epoch:0{max(1, n_digits)}d}.txt'])


# -----------------------------------------------------------------------------
#  Saving and restoring data
# -----------------------------------------------------------------------------
def save_pytree(fpath, tree):
    with open(f'{fpath}_arrays.npy', 'wb') as f:
        for x in jax.tree_util.tree_leaves(tree):
             np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, tree)

    with open(f'{fpath}_tree.pkl', 'wb') as f:
        pickle.dump(tree_struct, f)


def save_dicts(fpath, tree_dict):
    """
    Save pytrees given in tree_dict in folder 'fpath'.

    The keys of 'tree_dict' will be used as file name, the corresponding values
    (pytrees) will be saved into .npy files.
    """
    for k, v in tree_dict.items():
        save_pytree(f'{fpath}_{k}', v)


def save(settings, tree_dict, fname=None, check_save_interval=True):
    epoch = settings['workings']['epoch']
    save_interval = settings['output']['params_save_interval']

    if check_save_interval is True and epoch % save_interval != 0:
        return

    saved_epochs = settings['workings'].get('saved_epochs', list())
    max_entries = settings['output']['n_params_to_keep']
    save_path = settings['workings']['save_path']

    if max_entries > 0 and len(saved_epochs) + 1 > max_entries:
        # remove oldest entries (normally just one set of files)
        n_del = len(saved_epochs) + 1 - max_entries
        del_epochs = saved_epochs[:n_del]

        saved_files = os.listdir(save_path)

        saved_files = [
            f for f in saved_files
            if f.split('_')[0].isdigit()
            and int(f.split('_')[0]) in del_epochs]

        for file in saved_files:
            os.remove(os.path.join(save_path, file))

    if fname is None:
        n_digits = len(str(settings['training']['max_epochs']))
        fname = f'{epoch:0{max(1, n_digits)}d}'

    fpath = os.path.join(save_path, fname)
    save_dicts(fpath, tree_dict)
    saved_epochs.append(epoch)
    settings['workings']['saved_epochs'] = saved_epochs

    evaluate(settings, fpath)


def restore_pytree(fpath):
    with open(f'{fpath}_tree.pkl', 'rb') as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)

    with open(f'{fpath}_arrays.npy', 'rb') as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


def restore_dicts(fpath, tree_list):
    tree_dict = dict()

    for k in tree_list:
        tree_dict[k] = restore_pytree(f'{fpath}_{k}')
        
    return tree_dict


# -----------------------------------------------------------------------------
#  Visualization
# -----------------------------------------------------------------------------
def get_epochs_results(fpath, files):
    epochs = np.empty(len(files))
    results = np.empty(len(files))
    n_games = np.empty(len(files), dtype=int)

    for fid, file in enumerate(files):
        epochs[fid] = int(file[:-4].split('_')[1])

        with open(os.path.join(fpath, file), 'r') as f:
            contents = f.read()

        contents = contents.split(' ')
        results[fid] = float(contents[1])
        n_games[fid] = int(contents[0][:-1])  # -1 to remove ':'

    n_games = np.unique(n_games)

    if len(n_games) != 1:
        print(f'Warning: Evaluations used different numbers of games: {n_games}.')

    else:
        n_games = n_games[0]

    sort_inds = np.argsort(epochs)
    return epochs[sort_inds], results[sort_inds], n_games


def plot_evaluation_results_opponent(fpath, files, opponent, ax=None):
    epochs, results, n_games = get_epochs_results(fpath, files)

    if ax is None:
        plt.plot(epochs, results, marker='o', label=opponent)
        plt.title(f'{n_games} Games')
        plt.xlabel('Epochs')
        plt.ylabel('% Wins of Player 1')

        yb, yt = plt.ylim()
        yb, yt = min(yb-0.01, 0.49), max(yt+0.01, 1.01)
        plt.ylim(yb, yt)

        plt.axhline(1.0, linestyle='--', color='black', lw=0.8)
        plt.axhline(0.5, linestyle='--', color='black', lw=0.8)
        plt.legend()
        plt.show()

    else:
        fname = os.path.basename(os.path.dirname(fpath))
        ax.plot(epochs, results, marker='o', label=fname+' vs '+opponent)


def plot_evaluation_results(fpath, ax=None):
    # fpath should contain a directory called 'evaluation'
    fpath = os.path.join(fpath, 'evaluation')
    files = os.listdir(fpath)

    # files have format 'epoch_{player}_vs_{ref_opponent}.txt'
    opponents = np.unique([f[:-4].split('_vs_')[-1] for f in files])

    for opponent in opponents:
        files_ = [f for f in files if f[:-4].split('_vs_')[-1] == opponent]
        plot_evaluation_results_opponent(fpath, files_, opponent, ax)


# -----------------------------------------------------------------------------
#  Online-Algorithm for computing mean and variance (from Wikipedia)
# -----------------------------------------------------------------------------
# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def mean_var_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


# Retrieve the mean, variance and sample variance from an aggregate
def mean_var_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return (float("nan"),) * 3
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
