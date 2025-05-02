import sys
import numpy as np
import random

from simulator import human_game
from settings.machine_functions import machine_fns


if __name__ == '__main__':
    assert len(sys.argv) == 2, \
        'You need to provide the number of games.'

    n_games = int(sys.argv[1])

    print('If you want to play against a machine use one of the '
          'following options as player name:')

    for name in machine_fns:
        print(' -', name)

    n0 = input('\nName of player 0: ')
    n1 = input('Name of player 1: ')
    seed = input('Seed (empty uses current time): ')

    if seed == '':
        seed = None

    else:
        ss = np.random.SeedSequence(seed)


    names = (n0, n1)

    player_types = (
        n0 if n0 in machine_fns else 'human',
        n1 if n1 in machine_fns else 'human')

    wins = [0, 0]
    flip = random.uniform(0, 1) > 0.5

    if flip is True:
        names = names[::-1]
        player_types = player_types[::-1]

    print(f'\nPlaying to best of {n_games} games.')

    for i in range(n_games):
        print(f'Starting Game {i+1}/{n_games}')

        winner = human_game.play(
            seed=None if seed is None else ss.spawn(1)[0],
            names=names, player_types=player_types)

        if winner is None:
            print('Aborted Match.')
            sys.exit()

        if flip is True:
            winner = (winner + 1) % 2

        wins[winner] += 1

        if flip is True:
            print(f'Score: {names[1]} {wins[0]} : {wins[1]} {names[0]}')

        else:
            print(f'Score: {names[0]} {wins[0]} : {wins[1]} {names[1]}')

        if any(w > n_games / 2 for w in wins):
            break

        names = names[::-1]
        player_types = player_types[::-1]
        flip = not flip

    if wins[0] == wins[1]:
        print('Game ended in a draw.')
        sys.exit()

    if flip is True:
        names = names[::-1]
        player_types = player_types[::-1]

    winner = wins[1] > wins[0]
    print(f'Congratulations {names[winner]}, you won!')
