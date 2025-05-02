import time
import copy
import numpy as np
import jax.numpy as jnp


# helper for testing code
def pprint_state(state, i=0):  # Todo: improve alignment and spaces
    n1 = [f'{j:3}' for j in state[i, 7:13][::-1]]
    n0 = [f'{j:3}' for j in state[i, :6]]
    print('-----------------------------------')
    print('|   |', '|'.join(n1), '|   |')
    print(f'|  {state[i, -1]}|-------------------------|  {state[i, 6]}|')
    print('|   |', '|'.join(n0), '|   |')
    print('-----------------------------------')
    # print(f'{state[i, -1]}  {state[i, 7:13][::-1]}  {state[i, 6]}\n   {state[i, :6]}')


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


def get_initial_state(n_games):
    # state will also be used for the RL model, format:
    # - each entry represents the number of stones in one of the squares
    state = np.zeros((n_games, 14), dtype=int)
    state[:, :6] = 4
    state[:, 7:13] = 4
    return state


def get_options(state, current_player):
    # options are just the non-zero entries of the player's pits
    n_games = len(state)
    offset = 7 * current_player  # shape: (n_games,) 

    inds = np.repeat(np.arange(6)[None, :], n_games, axis=0)
    inds += offset[:, None]

    n_stones = np.take_along_axis(state, inds, axis=-1)
    inds = np.where(n_stones==0, -1, inds)

    return inds


def flip_indices(inds, mirror=False):
    inds = np.array(inds)
    mask = (inds == -1)

    if mirror is True:
        inds = 12 - inds
        inds = np.where(np.logical_or(inds == -1, inds == 6), inds + 7, inds)

    else:
        inds = (inds + 7) % 14

    inds[mask] = -1
    return inds


def flip_board(state):  # used by machine_agent.py
    new_state = state.copy()
    new_state[..., :7] = state[..., 7:]
    new_state[..., 7:] = state[..., :7]
    return new_state


def update_state(state, current_player, game_winner, options, idx):
    # state: (n_games, 14)
    # current_player: (n_games,)
    # options: (n_games, 6)
    # idx: (n_games,)
    # print('idx:', idx)
    # pprint_state(state, 0)

    # shape (3, 1)
    n_stones = np.take_along_axis(state, idx[:, None], axis=-1)
    n_skips = (idx[:, None] + n_stones - 7*current_player[:, None]) // 13
    replay = np.zeros(len(state), dtype=bool)
    store_idx = 6 + 7 * current_player
    # print(f'n_stones: {n_stones}, n_skips={n_skips}, replay={replay}, store_idx={store_idx}')

    # pick up all stones from selected index
    np.put_along_axis(state, idx[:, None], 0, axis=-1)

    # add stones to pits (Todo: avoid loop somehow)
    for i in range(len(state)):
        # print('GAME', i, ', winner:', game_winner[i])

        if game_winner[i] != -1:
            continue

        inds = np.arange(
            idx[i] + 1,
            idx[i] + n_stones[i, 0] + n_skips[i, 0] + 1)
        # print(' inds:', inds)
        inds = inds % 14
        state[i, inds] += 1

        if inds[-1] == store_idx[i]:
            replay[i] = True

        else:
            last_idx = inds[-1]
            n_stones_p = state[i, last_idx]
            n_stones_o = state[i, flip_indices(last_idx, mirror=True)]
            lims = (0, 6) if current_player[i] == 0 else (7, 13)

            if n_stones_p == 1 and n_stones_o > 0 and last_idx >= lims[0] and last_idx < lims[1]:
                state[i, store_idx[i]] += n_stones_p + n_stones_o
                state[i, last_idx] = 0
                state[i, flip_indices(last_idx, mirror=True)] = 0
        # pprint_state(state, i)

    # remove stones added to opponent's store
    opp_store_idx = flip_indices(store_idx, mirror=True)
    n_store = np.take_along_axis(state, opp_store_idx[:, None], axis=-1) - n_skips
    np.put_along_axis(state, opp_store_idx[:, None], n_store, axis=-1)

    return state, replay


def update_game_winner(state, game_winner):
    count_0 = state[:, :6].sum(axis=-1)
    count_1 = state[:, 7:13].sum(axis=-1)

    finished_0 = (count_0 == 0)
    finished_1 = (count_1 == 0)

    finished_mask = np.logical_or(finished_0, finished_1)
    mask = np.logical_and(game_winner == -1, finished_mask)

    state[:, 6] += np.where(mask, count_0, 0)
    state[:, 13] += np.where(mask, count_1, 0)

    state[:, :6] = np.where(mask[:, None], 0, state[:, :6])
    state[:, 7:13] = np.where(mask[:, None], 0, state[:, 7:13])

    game_winner[mask] = (state[:, 13] > state[:, 6])[mask]

    # there could be a draw
    game_winner[mask] = np.where(
        state[mask][:, 13] == state[mask][:, 6], 0.5, game_winner[mask])

    return state, game_winner


# need to save for every move: board, options, idx
# need to save once: winner
def play(machine_fns, seed=None, settings=None):
    n_games = 1

    if settings is not None:
        n_games = settings['sample_creation'].get(
            'n_games_per_sample_creator', 1)

    # print('machine_game play, n_games =', n_games)
    state = get_initial_state(n_games)

    if seed is None:
        seed = int(time.time())

    # print('machine_game play, seed =', seed)
    rng = np.random.default_rng(seed)
    current_player = np.zeros(n_games, dtype=int)
    game_winner = -np.ones(n_games, dtype=int)

    # prepare lists to save data, don't know in advance how many moves there
    # will be per game, can't preallocate contiguous memory ...
    _board, _options, _idx = list(), list(), list()

    for _ in range(2 * n_games):  # '2 *' because we save data for each player separately
        _board.append(list())
        _options.append(list())
        _idx.append(list())

    # play until last game has finished
    # mid = 0  # temp
    while True:
        # print(f'\nnew move {mid}, player_ids={current_player[:2]}')
        # mid += 1  # temp
        options = get_options(state, current_player)
        # print(' options:', options[:2])
        idx = machine_fns[0](state, options, current_player, rng)
        # print(' idx:', idx[:2])

        if machine_fns[1] is not machine_fns[0]:
            idx_1 = machine_fns[1](state, options, current_player, rng)
            idx = np.where(current_player, idx_1, idx)

        # print('idx:', idx.shape)

        # save data
        for gid, (s, o, i, p) in enumerate(zip(state, options, idx,
                                               current_player)):

            if game_winner[gid] == -1 and i == -1:  # Temp: just to ensure code works as intended
                raise RuntimeError('...')

            # game has finished --> don't need to save anymore data
            if game_winner[gid] != -1:
                continue

            gid = 2*gid+p
            _board[gid].append(copy.deepcopy(s))
            _options[gid].append(copy.deepcopy(o))
            _idx[gid].append(i)

        """
        import contextlib
        gid = 1
        if game_winner[gid] == -1:
            for j in range(2*gid, 2*gid+2):
                print(' j =', j)
                with contextlib.suppress(IndexError):
                    print(' _board:', _board[j][-1])
                with contextlib.suppress(IndexError):
                    print(' _options:', _options[j][-1])
                with contextlib.suppress(IndexError):
                    print(' _idx:', _idx[j][-1])
        """

        # move and update variables
        state, reroll = update_state(state, current_player, game_winner, options, idx)
        # print('state_post:\n', state)
        # print('reroll:', reroll)
        current_player = (current_player + np.where(reroll, 0, 1)) % 2
        state, game_winner = update_game_winner(state, game_winner)
        # print('game_winners:', game_winner)

        # for i in range(len(state)):
            # print(f'\nGame {i}:')
            # pprint_state(state, i=i)

        if (game_winner != -1).all():
            break

    return _board, _options, _idx, np.repeat(game_winner, 2)
