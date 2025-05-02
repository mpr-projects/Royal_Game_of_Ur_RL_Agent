import sys
import time
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.backend_bases import MouseButton

from simulator.game_manager import GameManager
from simulator.utils import convert_to_idx
from simulator.default_settings import (
    SHOW_SQUARE_INDICES, COLORS, BG, REROLL_INDICES)

import settings.machine_functions



# -----------------------------------------------------------------------------
# Functions for plotting human games
# -----------------------------------------------------------------------------
def get_number(n, bg=None):
    bg = BG if bg is None else bg

    if n > 9:
        n1 = get_number(n//10, bg)
        n_pad = bg * np.ones((n1.shape[0], 1))
        n2 = get_number(n%10, bg)
        return np.concatenate((n1, n_pad, n2), axis=1)

    res = bg * np.ones((5, 3))

    if n in [0, 1, 2, 3, 4, 7, 8, 9]:
        res[3:, -1] = 1

    if n in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
        res[:3, -1] = 1

    if n in [0, 4, 5, 6, 8, 9]:
        res[3:, 0] = 1

    if n in [0, 2, 6, 8]:
        res[:3, 0] = 1

    if n in [0, 2, 3, 5, 6, 7, 8, 9]:
        res[-1] = 1

    if n in [2, 3, 4, 5, 6, 8, 9]:
        res[2] = 1

    if n in [0, 2, 3, 5, 6, 8, 9]:
        res[0] = 1

    return res
    

def get_square_start(game, idx):
    player_id = int(idx != 4)  # refers to id of player on this side of the board
    img = np.zeros((36, 36))
    n_left = game.n_remaining_stones(player_id)

    for i in range(n_left):
        s  = 2+i*4
        img[s:s+2, 2:4] = 1 if player_id == 0 else 0.7

    if SHOW_SQUARE_INDICES is False and player_id == 0:
        img = img[::-1]

    return img[::-1]

def get_square_end(game, idx):
    player_id = int(idx != 5)  # refers to id of player on this side of the board
    img = np.zeros((36, 36))
    n_finished = game.n_finished_stones(player_id)

    for i in range(n_finished):
        s  = 2+i*4
        img[s:s+2, -4:-2] = 1 if player_id == 0 else 0.7

    if SHOW_SQUARE_INDICES is False and player_id == 0:
        img = img[::-1]

    img = img[::-1]

    if SHOW_SQUARE_INDICES is True:
        n = get_number(idx, bg=0)
        img[2:2+n.shape[0], -2-n.shape[1]:-2] = n

    return img

def get_square(game, idx):
    if idx in [4, 20]:
        return get_square_start(game, idx)

    if idx in [5, 21]:
        return get_square_end(game, idx)

    img = BG * np.ones((36, 36))

    if idx in REROLL_INDICES:
        img[13:-13, 5:-5] = 0.3
        img[5:-5, 13:-13] = 0.3

    # frame
    img[0] = 1
    img[-1] = 1
    img[:, 0] = 1
    img[:, -1] = 1

    # idx
    if SHOW_SQUARE_INDICES is True:
        n = get_number(idx)
        img[2:2+n.shape[0], -2-n.shape[1]:-2] = n

    # markers
    if game.has_stone_at_index(player_id=0, idx=idx):
        img[10:-10, 10:-10] = np.maximum(img[10:-10, 10:-10], 1)

    elif game.has_stone_at_index(player_id=1, idx=idx):
        img[10:-10, 10:-10] = np.maximum(img[10:-10, 10:-10], 0.7)

    return img


def set_title(game, human_game_state):
    names = human_game_state['names']

    if game.has_finished:
        winner = game.winner
        human_game_state['labels'][winner].set_text(f'Congratulations {names[winner]}, you won!')
        human_game_state['labels'][(winner+1)%2].set_text('')
        return

    player_id = game.current_player
    text = f'{names[player_id]} ({COLORS[player_id]}), rolled {game.roll}'

    if game.n_options == 0:
        text += ' (click anywhere)'

    human_game_state['labels'][player_id].set_text(text)
    human_game_state['labels'][(player_id+1)%2].set_text('')


def get_board(game):
    img = BG * np.ones((108, 288))

    for i in range(24):
        r, c = i // 8, i % 8 
        img[r*36:(r+1)*36, c*36:(c+1)*36] = get_square(game, i)

    return img


def update_plot(game, human_game_state):
    set_title(game, human_game_state)
    human_game_state['im'].set_data(get_board(game))
    plt.draw()


# -----------------------------------------------------------------------------
# Human-Human or Human-Machine Play
# -----------------------------------------------------------------------------
def on_click_human(event, **kwargs):
    r, c = event.ydata // 36, event.xdata // 36
    return convert_to_idx(r, c)


def get_on_click_machine(machine_fn_name):
    get_machine_fn, args, kwargs = \
        settings.machine_functions.machine_fns[machine_fn_name]

    machine_fn = get_machine_fn(*args, **kwargs)

    def on_click_machine(game, **kwargs):
        return machine_fn(game)

    return on_click_machine


def get_on_click(game, on_click_fns, human_game_state):
    def on_click(event):
        if game.has_finished:
            game.print_roll_history(human_game_state['names'])
            plt.close()
            return

        if event.button is not MouseButton.LEFT:
            return

        if game.n_options == 0:
            game.move(game.PASS)
            update_plot(game, human_game_state)
            return

        if not event.inaxes:
            return
    
        idx = on_click_fns[game.current_player](game=game, event=event)

        if idx not in game.options:
            return

        game.move(idx)
        update_plot(game, human_game_state)
    return on_click


def play(seed=None, names=('Player 0', 'Player 1'), player_types=('human', 'human')):
    if seed is None:
        seed = int(time.time())

    game = GameManager(seed)
    game.start()
    board = get_board(game)

    human_game_state = dict(
        names=names,
        im=plt.imshow(board, cmap='Greens', origin='lower'))

    ax = plt.gca()
    ax.axis('off')

    text0 = ax.annotate(
        '', xy=(0.5, -0.05), xycoords='axes fraction', xytext=(0, -30),
        textcoords='offset pixels',  ha='center', fontsize=12, va='bottom')

    text1 = ax.annotate(
        '', xy=(0.5, 1.05), xycoords='axes fraction', xytext=(0, +30),
        textcoords='offset pixels',  ha='center', fontsize=12, va='top')

    human_game_state['labels'] = [text0, text1]
    set_title(game, human_game_state)
    
    on_click_fns = (
        on_click_human if player_types[0] == 'human' else get_on_click_machine(player_types[0]),
        on_click_human if player_types[1] == 'human' else get_on_click_machine(player_types[1]))

    on_click = get_on_click(game, on_click_fns, human_game_state)
    plt.connect('button_press_event', on_click)
    plt.show()

    with contextlib.suppress(AttributeError):
        return game.winner
