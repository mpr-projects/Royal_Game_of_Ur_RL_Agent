import time
import copy
import numpy as np
import jax.numpy as jnp

from .default_settings import DEFAULT_N_STONES, REROLL_INDICES


# used in evaluation
def machine_fn_random(state, options, current_player, rng):
    # this is a bit tricky because each game can have a different
    # number of options
    n_games = len(options)
    idx = np.zeros(n_games, dtype=int)

    for i in range(n_games):
        o = options[i]
        o = o[o != -1]

        if len(o) != 0:
            idx[i] = rng.choice(o)

    return idx


def get_initial_state(n_games, N_STONES):
    # state will also be used for the RL model, format:
    # - player 0 placed stones
    # - player 0 remaining stones
    # - dice roll
    # - player 1 remaining stones
    # - player 1 placed stones
    # note, this can easily flipped to the view of player 1 by reversing
    # the first and last dimensions
    state = np.zeros((n_games, 3, 8, 5), dtype=int)
    state[:, :, :, 1] = N_STONES
    state[:, :, :, 3] = N_STONES

    return state


def roll_dice(rng, n_games):
    """ Rolls the dice in the same way as in the board game. """
    p = rng.random(n_games)
    roll = np.zeros(n_games)

    roll[np.logical_and(p >= 0.0650, p < 0.3125)] = 1
    roll[np.logical_and(p >= 0.3125, p < 0.6875)] = 2
    roll[np.logical_and(p >= 0.6875, p < 0.9375)] = 3
    roll[p >= 0.9375] = 4

    return roll


def get_prev_rc(player_id, r, c, roll):
    """ Find row/col values before moving by 'roll'. """
    in_protected_row = (r == 2 * player_id)
    in_middle_row = np.logical_not(in_protected_row)

    c += np.where(in_protected_row, roll, 0)
    left_protected_row = (c >= 8)
    r = np.where(np.logical_and(in_protected_row, left_protected_row), 1, r)
    c = np.where(np.logical_and(in_protected_row, left_protected_row), 15-c, c)

    c -= np.where(in_middle_row, roll, 0)
    left_middle_row = (c < 0)
    r = np.where(np.logical_and(in_middle_row, left_middle_row), 2*player_id, r)
    c = np.where(np.logical_and(in_middle_row, left_middle_row), -c-1, c)

    return r.astype(int), c.astype(int)


def convert_to_idx(r, c):
    return (8 * r + c).astype(int)


def convert_to_rc(idx):
    """ Converts from the 1d board index to row/column indices. """
    return idx // 8, idx % 8


def get_options(state, current_player, N_STONES):
    n_games = len(state)
    roll = state[:, 0, 0, 2]
    zero_roll = (roll == 0)

    # extra column is required for when all seven options are used, to not
    # overwrite the last option (or not get an index error)
    options = -np.ones((n_games, N_STONES+1)).astype(int)
    oids = np.zeros(n_games).astype(int)

    # go through each square and check if a stone can be placed
    for sid in range(24):
        # start squares, can never move there
        if sid in [4, 20]:
            continue

        r, c = convert_to_rc(sid)
        sid_state = state[:, r, c]
        player_col = 4 * current_player  # 0 for player 0, 4 for player 1
        # print(f'sid={sid}, r={r}, c={c}')

        # can't move here if there's already a stone by the player
        contains_player_stone = np.take_along_axis(
                sid_state, player_col[:, None], axis=-1)[:, 0]

        # can move here if the square is 'roll'-distance from another stone of
        # the player
        prev_r, prev_c = get_prev_rc(current_player, r, c, roll)
        prev_idx = convert_to_idx(prev_r, prev_c)

        state_ = state.reshape((n_games, 24, 5))
        prev_state = np.take_along_axis(
            state_, prev_idx[:, None, None], axis=1)[:, 0]

        can_move_from_previous_index = np.take_along_axis(
            prev_state, player_col[:, None], axis=-1)[:, -1]

        # if the player has stones left and the square is 'roll'-distance from
        # the start then we can move here
        player_col = 1 + 2 * current_player  # 1 for player 0, 3 for player 1
        player_has_stones_remaining = np.take_along_axis(
                sid_state, player_col[:, None], axis=-1)[:, 0] > 0

        start_idx = np.where(current_player, 20, 4)
        sid_in_range_of_start = (prev_idx == start_idx)

        can_move_from_start = np.logical_and(
            player_has_stones_remaining,
            sid_in_range_of_start)

        can_move_to_sid = np.logical_or(
            can_move_from_previous_index,
            can_move_from_start)

        square_is_available = np.logical_and(
            np.logical_not(contains_player_stone),
            can_move_to_sid)

        # an opponent's stone on square 11 can't be removed
        if sid == 11:
            opponent_col = 4 * (1 - current_player)

            contains_opponent_stone = np.take_along_axis(
                sid_state, opponent_col[:, None], axis=-1)[:, 0]

            square_is_available = np.logical_and(
                square_is_available,
                np.logical_not(contains_opponent_stone))

        # protected rows are not reachable by the opponent
        if sid < 8:
            square_is_available[current_player == 1] = False

        if sid > 15:
            square_is_available[current_player == 0] = False

        # can't jump over the start/end squares (ie in protected rows [0, 2]
        # for squares just after the start the previous square cannot be just
        # before the finish)
        mask = np.logical_and(
            r != 1,
            np.logical_and(c < 4, prev_c > 5))

        square_is_available[mask] = False

        # if we rolled a zero then there can be no options
        square_is_available[zero_roll] = False  

        # update the appropriate entries in 'options'
        # print(f'square {sid} is available:', square_is_available.shape, square_is_available)

        mask = np.zeros_like(options, dtype=bool)
        np.put_along_axis(mask, oids[:, None], True, axis=1)
        mask = np.logical_and(mask, square_is_available[:, None])

        options[mask] = sid
        oids += np.where(square_is_available, 1, 0)
        # print('oids:', oids)

    return options[:, :-1] 


def update_state(state, current_player, options, idx):
    # - player 0 placed stones
    # - player 0 remaining (unplaced) stones
    # - dice roll
    # - player 1 remaining (unplaced) stones
    # - player 1 placed stones
    n_games = len(state)
    roll = state[:, 0, 0, 2]
    state_ = state.reshape((n_games, 24, 5))

    r, c = convert_to_rc(idx)
    prev_r, prev_c = get_prev_rc(current_player, r, c, roll)
    prev_idx = convert_to_idx(prev_r, prev_c)

    no_move = (options == -1).all(axis=-1)
    new_stone = np.isin(prev_idx, [4, 20])
    existing_stone = np.logical_not(new_stone)
    finished_stone = np.isin(idx, [5, 21])

    mask_idx = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask_idx, idx[:, None, None], True, axis=1)

    # add new stone of player
    player_col = 4 * current_player  # 0 for player 0, 4 for player 1

    mask = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask, player_col[:, None, None], True, axis=-1)

    mask = np.logical_and(mask, mask_idx)
    mask[no_move] = False
    mask[finished_stone] = False

    state_[mask] = 1

    # remove previous location of stone
    mask = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask, player_col[:, None, None], True, axis=-1)

    prev_mask_idx = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(prev_mask_idx, prev_idx[:, None, None], True, axis=1)

    mask = np.logical_and(mask, prev_mask_idx)
    mask[no_move] = False
    mask[new_stone] = False  # this is not really necessary (there is no stone at the start index anyway)

    state_[mask] = 0

    # remove opponent stone
    opponent_col = 4 * (1 - current_player)

    mask = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask, opponent_col[:, None, None], True, axis=-1)

    mask = np.logical_and(mask, mask_idx)

    # used below, column here is different to below, need to adjust by removing
    # the last axis
    opponent_has_stone = np.logical_and(mask, (state_ > 0)).any(axis=(1, 2))

    state_[mask] = 0  # remove opponent stone (if there was one)

    # increase number of remaining (unplaced) opponent stones
    opponent_col = 1 + 2 * (1 - current_player)

    mask = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask, opponent_col[:, None, None], True, axis=-1)

    mask = np.logical_and(mask, opponent_has_stone[..., None, None])
    state_[mask] += 1

    # reduce number of remaining (unplaced) stones
    player_col = 1 + 2 * current_player  # 1 for player 0, 3 for player 1

    mask = np.zeros_like(state_, dtype=bool)
    np.put_along_axis(mask, player_col[:, None, None], True, axis=-1)

    mask[no_move] = False
    mask[existing_stone] = False

    state_[mask] -= 1

    # check if a player can roll again
    reroll = np.isin(idx, REROLL_INDICES)
    reroll[no_move] = False

    return state.reshape((n_games, 3, 8, 5)), reroll


def update_game_winner(state, game_winner):
    mask = (game_winner == -1)
    winner_0 = (state[..., :2] == 0).all(axis=(1, 2, 3))
    game_winner[mask] = np.where(winner_0[mask], 0, -1)

    mask = (game_winner == -1)
    winner_1 = (state[..., -2:] == 0).all(axis=(1, 2, 3))
    game_winner[mask] = np.where(winner_1[mask], 1, -1)

    return game_winner


# need to save for every move: board, options, idx
# need to save once: winner

def play(machine_fns, seed=None, settings=None):
    """ Play multiple games in parallel. """

    # get initial board, stones, ...
    N_STONES = DEFAULT_N_STONES

    if settings is not None:
        if 'game' in settings:
            N_STONES = settings['game'].get('N_STONES', DEFAULT_N_STONES)

    n_games = 1

    if settings is not None:
        n_games = settings['sample_creation'].get(
            'n_games_per_sample_creator', 1)

    state = get_initial_state(n_games, N_STONES)

    if seed is None:
        seed = int(time.time())

    rng = np.random.default_rng(seed)
    current_player = np.zeros(n_games, dtype=int)
    game_winner = -np.ones(n_games, dtype=int)

    # prepare lists to save data, don't know in advance how many moves there
    # will be per game, can't preallocate contiguous memory ...
    _board, _options, _idx = list(), list(), list()

    for _ in range(2 * n_games):
        _board.append(list())
        _options.append(list())
        _idx.append(list())

    # play until last game has finished
    while True:
        # print(f'\nnew move, player_ids={current_player}')
        roll = roll_dice(rng, n_games)
        state[..., 2] = roll[:, None, None]
        options = get_options(state, current_player, N_STONES)
        idx = machine_fns[0](state, options, current_player, rng)

        """
        print('roll:', roll)
        print('shapes:', roll.shape, state.shape)
        print('options:', options)
        print('selected:', idx)
        """

        if machine_fns[1] is not machine_fns[0]:
            idx_1 = machine_fns[1](state, options, current_player, rng)
            idx = np.where(current_player, idx_1, idx)

        # save data
        for gid, (s, o, i, p) in enumerate(zip(state, options, idx,
                                               current_player)):
            # no options --> no move
            if (o == -1).all():
                continue

            if i == -1:  # Temp: just to ensure code works as intended
                raise RuntimeError('...')

            # game has finished --> don't need to save anymore data
            if game_winner[gid] != -1:
                continue

            gid = 2*gid+p
            _board[gid].append(copy.deepcopy(s))
            _options[gid].append(copy.deepcopy(o))
            _idx[gid].append(i)

        # move and update variables
        state, reroll = update_state(state, current_player, options, idx)
        # print('state_post:\n', state)
        # print('reroll:', reroll)
        current_player = (current_player + np.where(reroll, 0, 1)) % 2
        game_winner = update_game_winner(state, game_winner)
        # print('game_winners:', game_winner)

        if (game_winner != -1).all():
            break

    """
    print('_board:', [len(b) for b in _board])
    print('_options:', [len(b) for b in _options])
    print('_idx:', [len(b) for b in _idx])
    print('game_winners:', game_winner)
    """
    return _board, _options, _idx, np.repeat(game_winner, 2)
