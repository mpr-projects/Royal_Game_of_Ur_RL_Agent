# this file will be symlinked in the respective game-folders, running a
# symlink'ed file uses the path of the target file, not of the link file
import os
import sys
sys.path.remove(sys.path[0])
sys.path.append(os.path.dirname(__file__))

import yaml
import argparse
import contextlib
import multiprocessing
import numpy as np

import training._utils
import interface._simulator_interface
import simulator.machine_game



def test_model(player1, n_games, device='cpu', player2='random', fname=None):
    """ player1 and player2 can also be 'random'; result will be written to folder 'evaluate' in folder of player1 """
    if fname is None:
        fname = 'results_against_random.txt'

    machine_fns = list()

    for player in [player1, player2]:
        if player == 'random':
            machine_fns.append(simulator.machine_game.machine_fn_random)
            continue

        p, e = os.path.split(player)

        machine_fns.append(
            interface._simulator_interface.get_machine_fn_rl(
                p, epoch=int(e), backend=device, print_probs=False))

    settings = {
        'sample_creation': {'n_games_per_sample_creator': n_games//2}}

    winner_0 = simulator.machine_game.play(
        machine_fns, seed=5, settings=settings)[-1][::2]

    machine_fns = [machine_fns[1], machine_fns[0]]
    settings = {
        'sample_creation': {'n_games_per_sample_creator': n_games - n_games//2}}

    winner_1 = simulator.machine_game.play(
        machine_fns, seed=6, settings=settings)[-1][::2]
    winner_1 = (winner_1 + 1) % 2

    winner_list = np.concatenate((winner_0, winner_1))
    result = 1 - sum(winner_list) / len(winner_list)

    p = os.path.join(os.path.dirname(player1), 'evaluation')
    os.makedirs(p, exist_ok=True)

    with open(os.path.join(p, fname), 'w') as f:
        f.write(f'{n_games}: {result} (win fraction of player 1 vs player 2)\n')

    return result


def load_settings_from_file(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def get_available_players(folder):
    # returns all available players (saved parameters) sorted last first
    av_players = set()

    for f in os.listdir(folder):
        epoch = f.split('_')[0]

        if not epoch.isdigit():
            continue

        av_players.add(epoch)

    av_players = list(av_players)
    av_players.sort(reverse=True)
    return av_players


def parse_settings_from_file(settings):
    n_games = settings['n_games']
    n_players = settings['n_players']
    device = settings['backend']

    ref_folder = settings['folder_reference_player']
    evl_folder = settings.get('folder_evaluated_player', ref_folder)

    ref_player = settings['reference_player']

    if ref_player == 'last':
        ref_player = os.path.join(
            ref_folder, get_available_players(ref_folder)[0])

    evl_player = settings['evaluated_player']

    if evl_player == 'all':
        evl_player = [os.path.join(evl_folder, p)
                      for p in get_available_players(evl_folder)]

    if not isinstance(evl_player, list):
        evl_player = [evl_player]

    with contextlib.suppress(ValueError):
        evl_player.remove(ref_player)

    return evl_player, n_games, device, ref_player, None, n_players


def parse_arguments():
    parser = argparse.ArgumentParser(prog='EvaluateMachinePlayers')

    parser.add_argument('--settings_folder', help=(
        'If given then file "evaluation_settings.yaml" in the folder will be'
        ' used and the remaining settings will be ignored.'))

    parser.add_argument('--evaluated_player')
    parser.add_argument('--n_games', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reference_player', default='random')
    parser.add_argument('--fname')

    args = parser.parse_args()

    if args.settings_folder is not None:
        settings = load_settings_from_file(
            os.path.join(args.settings_folder, 'evaluation_settings.yaml'))

        return parse_settings_from_file(settings)

    assert args.evaluated_player is not None and args.n_games is not None
    return [args.evaluated_player], args.n_games, args.device, args.reference_player, args.fname, 1


def get_p_ref(path_ref):
    if path_ref == 'random':
        return 'random'

    p, epoch = os.path.split(path_ref)
    _, fname = os.path.split(p)
    return '-'.join([fname, epoch])


def pool_initializer(n_games_, device_, path_ref_, fname_):
    global n_games, device, path_ref, fname, p_ref
    n_games, device, path_ref, fname = n_games_, device_, path_ref_, fname_
    p_ref = get_p_ref(path_ref)


def test_parallel(player):
    print('Testing', player)
    p_evl = os.path.basename(player).split('_')[0]
    fname = f'epoch_{p_evl}_vs_{p_ref}.txt'
    fpath = os.path.join(os.path.dirname(player), 'evaluation', fname)

    if os.path.isfile(fpath):
        print(f'File {fname} exists. Skipping...')
        return

    test_model(player, n_games, device=device, player2=path_ref, fname=fname)


if __name__ == '__main__':
    path_evl, n_games, device, path_ref, fname, n_players = parse_arguments()

    if device == 'cpu':
        training._utils.initializer_cpu_only()

    if len(path_evl) == 1 and fname is not None:
        test_model(path_evl[0], n_games, device=device, player2=path_ref, fname=fname)
        sys.exit()

    p_ref = get_p_ref(path_ref)

    if n_players == 1:
        for player in path_evl:
            p_evl = os.path.basename(player).split('_')[0]
            fname = f'epoch_{p_evl}_vs_{p_ref}.txt'
            test_model(player, n_games, device=device, player2=path_ref, fname=fname)

        sys.exit()

    import time
    start = time.time()
    with multiprocessing.Pool(processes=n_players,
                              initializer=pool_initializer,
                              initargs=(n_games, device, path_ref, fname)) as pool:
        pool.map(test_parallel, path_evl, chunksize=2)
    print('time:', time.time()-start)
