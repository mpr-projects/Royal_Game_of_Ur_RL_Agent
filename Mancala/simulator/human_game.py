import numpy as np
import time

from simulator.machine_game import (
    play, get_initial_state, get_options, update_state, update_game_winner,
    pprint_state)

import settings.machine_functions


def machine_fn_human(state, options, current_player, rng):
    pprint_state(state, 0)
    while True:
        try:
            print('options:', options)
            idx = input('Enter idx: ')
            idx = np.array([int(i) for i in idx.split(' ') if int(i) in options])

            if len(idx) != len(options):
                print(f'Wrong number of indices ({len(idx)}),'
                      f' expected {len(options)}.')
                continue

            return idx

        except ValueError:
            print('Could not accept input.')
            pass


def get_machine_fn(machine_fn_name):
    gmf, args, kwargs = \
        settings.machine_functions.machine_fns[machine_fn_name]

    return gmf(*args, **kwargs)


def play(seed=None, names=('Player 0', 'Player 1'), player_types=('human', 'human')):
    print('palyer taypes:', player_types)
    n_games = 1
    state = get_initial_state(n_games)

    if seed is None:
        seed = int(time.time())

    rng = np.random.default_rng(seed)
    current_player = np.zeros(n_games, dtype=int)
    game_winner = -np.ones(n_games, dtype=int)

    player_fns = (
        machine_fn_human if player_types[0] == 'human'
        else get_machine_fn(player_types[0]),
        machine_fn_human if player_types[1] == 'human'
        else get_machine_fn(player_types[1]),
    )

    # play until last game has finished
    while True:
        print(f'\nNew Move, player={current_player[0]}')
        options = get_options(state, current_player)
        print('options:', options)
        idx = player_fns[current_player[0]](
            state, options, current_player, rng)

        if player_types[current_player[0]] != 'human':
            print('Turn of the computer. Current board:')
            pprint_state(state, 0)

        # save data
        for gid, (s, o, i, p) in enumerate(zip(state, options, idx,
                                               current_player)):

            if game_winner[gid] == -1 and i == -1:  # Temp: just to ensure code works as intended
                raise RuntimeError('...')

            # game has finished --> don't need to save anymore data
            if game_winner[gid] != -1:
                continue

        # move and update variables
        state, reroll = update_state(state, current_player, game_winner, options, idx)
        # print('state_post:\n', state)
        # print('reroll:', reroll)
        state, game_winner = update_game_winner(state, game_winner)
        # print('game_winners:', game_winner)

        if player_types[current_player[0]] != 'human':
            print(f'Computer played index {idx[0]}. New board:')
            pprint_state(state, 0)
            input('')

        current_player = (current_player + np.where(reroll, 0, 1)) % 2

        # for i in range(len(state)):
            # print(f'\nGame {i}:')
            # pprint_state(state, i=i)

        if (game_winner != -1).all():
            break

    return game_winner[0]


if __name__ == '__main__':
    winner = play([machine_fn_human, machine_fn_human])
    print('Congratulations player {winner}, you won!')
