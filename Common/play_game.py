# this file will be symlinked in the respective game-folders, running a
# symlink'ed file uses the path of the target file, not of the link file
import os
import sys
sys.path.remove(sys.path[0])
sys.path.append(os.path.dirname(__file__))

from simulator import human_game
from settings.machine_functions import machine_fns


if __name__ == '__main__':
    print('If you want to play against a machine use one of the '
          'following options as player name:')

    for name in machine_fns:
        print(' -', name)

    n0 = input('\nName of player 0: ')
    n1 = input('Name of player 1: ')
    seed = input('Seed (empty uses current time): ')

    seed = None if seed == '' else int(seed)

    player_types = (
        n0 if n0 in machine_fns else 'human',
        n1 if n1 in machine_fns else 'human')

    names = (n0, n1)

    winner = human_game.play(
        seed=seed, names=names, player_types=player_types)

    if winner is not None:
        print('winner:', names[winner], f'(player {winner})')
