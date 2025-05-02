import numpy as np

from .utils import roll_dice, get_next_idx, get_prev_idx
from .utils import convert_to_idx, convert_to_rc
from .default_settings import DEFAULT_N_STONES, REROLL_INDICES


def flip_board(board):
    # ... allow for additional batch dimension
    return board[..., ::-1, :, ::-1]


def flip_indices(indices):
    return indices - 16 * (indices // 8 - 1)


class GameManager:
    """
    Saves the moves of each player. Allows to go through the game backwards
    after it has finished. This is used by the RL model to compute updates.
    """

    # if 0 was rolled then no move is possible, use PASS instead of index
    PASS = -1

    def __init__(self, seed, settings=None):  # , player_fns, names=['Player 0', 'Player 1'], verbose=True):
        """
        If parameter 'settings' is given it should be a dict with entry
        'N_STONES'. If not given then the 'usual' value of 7 stones is used.
        will be used.
        """
        self._move_id = 0
        self._player_id = 0  # player 0 starts the game
        self._has_finished = False
        self._rng = np.random.default_rng(seed=seed)

        # history has four components:
        #  - player_id
        #  - (dice) roll
        #  - idx (square to move to)
        #  - removed_opponent_stone
        self._history = list()

        self.N_STONES = DEFAULT_N_STONES

        if settings is not None:
            self.N_STONES = settings.get('N_STONES', DEFAULT_N_STONES)

        # state will also be used for the RL model, format:
        # - player 0 placed stones
        # - player 0 remaining stones
        # - dice roll
        # - player 1 remaining stones
        # - player 1 placed stones
        # note, this can easily flipped to the view of player 1 by reversing
        # the first and last dimensions
        self._state = np.zeros((3, 8, 5), dtype=int)
        self._state[:, :, 1] = self.N_STONES
        self._state[:, :, 3] = self.N_STONES

    @property
    def move_id(self):
        return self._move_id

    @property
    def current_player(self):
        return self._player_id

    @property
    def n_options(self):
        return len(self._options)

    @property
    def options(self):  # return copy to prevent external changes
        return np.asarray(self._options)

    def get_n_moves(self, player_id='both'):
        assert player_id in ['both', 0, 1]
        if player_id == 'both':
            return len(self._history)
        return sum([1 for h in self._history if h[0] == player_id])

    @property
    def state(self):  # return copy to prevent external changes
        return np.array(self._state)

    @property
    def has_finished(self):
        return self._has_finished

    def n_placed_stones(self, player_id=None):
        player_id = self._player_id if player_id is None else player_id
        return len(self._player_stones if player_id == self._player_id
                   else self._opponent_stones)

    def n_remaining_stones(self, player_id=None):
        player_id = self._player_id if player_id is None else player_id
        return int(self._state[0, 0, 1+2*player_id])

    def n_finished_stones(self, player_id=None):
        player_id = self._player_id if player_id is None else player_id
        return (self.N_STONES
                -self.n_placed_stones(player_id)
                -self.n_remaining_stones(player_id))

    def start(self):
        self._prepare_turn()
        # at this point state and options contain all information required for ML model

    def move(self, idx):
        assert not self._has_finished
        self._process_move(idx)

        self._history.append((
            self._player_id,
            self._roll,
            idx,  # -1 for PASS, ie if roll was 0
            self._removed_opponent_stone))

        self._move_id += 1

        if self.n_finished_stones() == self.N_STONES:
            self._has_finished = True
            self.winner = self._player_id
            return

        # prepare state for next turn
        if idx not in REROLL_INDICES:  # those indices give a reroll
            self._player_id = (self._player_id + 1) % 2

        self._prepare_turn()

    def _prepare_turn(self):
        self._removed_opponent_stone = False
        self._roll = roll_dice(self._rng)
        self._state[:, :, 2] = self._roll
        self._player_stones = self._get_player_stone_indices()
        self._opponent_stones = self._get_opponent_stone_indices()
        self._options = self._get_options()

    @property
    def start_idx(self):
        return 4 if self._player_id == 0 else 20

    @property
    def target_idx(self):
        return 5 if self._player_id == 0 else 21

    def _get_player_stone_indices(self):
        r_inds, c_inds = (self._state[:, :, 4*self._player_id] == 1).nonzero()
        return [convert_to_idx(r, c) for r, c in zip(r_inds, c_inds)]

    def _get_opponent_stone_indices(self):
        r_inds, c_inds = (self._state[:, :, 4-4*self._player_id] == 1).nonzero()
        return [convert_to_idx(r, c) for r, c in zip(r_inds, c_inds)]

    def roll_overshoots(self, idx, next_idx):
        return (idx%8) > 5 and (next_idx%8) < 5

    def _get_options(self):
        options = list()
        
        if self._roll == 0:
            return options

        player_stones = self._player_stones
        opponent_stones = self._opponent_stones

        # check if a new stone can be placed
        if self.n_remaining_stones(self._player_id) > 0:
            next_idx = get_next_idx(self._player_id, self.start_idx, self._roll)

            if next_idx not in player_stones:
                options.append(next_idx)

        # check each existing stone
        for idx in player_stones:
            next_idx = get_next_idx(self._player_id, idx, self._roll)
            
            if next_idx in player_stones:
                continue  # already occupied by player

            if next_idx == 11 and next_idx in opponent_stones:
                continue  # protected square

            if self.roll_overshoots(idx, next_idx):
                continue

            options.append(next_idx)

        return options

    def set_player_stone(self, idx):
        r, c = convert_to_rc(idx)
        self._state[r, c, 4*self._player_id] = 1

    def set_opponent_stone(self, idx):
        r, c = convert_to_rc(idx)
        self._state[r, c, 4-4*self._player_id] = 1

    def remove_player_stone(self, idx):
        r, c = convert_to_rc(idx)
        self._state[r, c, 4*self._player_id] = 0

    def remove_opponent_stone(self, idx):
        r, c = convert_to_rc(idx)
        self._state[r, c, 4-4*self._player_id] = 0

    def change_player_stone_count(self, val):  # count of remaining stones
        self._state[:, :, 1+2*self._player_id] += val

    def change_opponent_stone_count(self, val):
        self._state[:, :, -2-2*self._player_id] += val

    def _process_move(self, idx):
        if idx == self.PASS:
            return

        assert idx in self._options, 'Invalid Move.'
        prev_idx = get_prev_idx(self._player_id, idx, self._roll)
        opponent_stones = self._opponent_stones

        if idx in opponent_stones:
            self._removed_opponent_stone = True
            self.remove_opponent_stone(idx)
            self.change_opponent_stone_count(+1)

        if prev_idx == self.start_idx:
            self.change_player_stone_count(-1)
        else:
            self.remove_player_stone(prev_idx)

        if idx != self.target_idx:
            self.set_player_stone(idx)

        # need updated to check if game is over
        self._player_stones = self._get_player_stone_indices()

    def _undo_move(self, idx):
        if idx == self.PASS:
            return

        self.remove_player_stone(idx)

        if self._removed_opponent_stone:
            self._removed_opponent_stone = False
            self.set_opponent_stone(idx)
            self.change_opponent_stone_count(-1)

        prev_idx = get_prev_idx(self._player_id, idx, self._roll)

        if prev_idx == self.start_idx:
            self.change_player_stone_count(+1)
        else:
            self.set_player_stone(prev_idx)

    # allow moving through the game history, this is used by the RL
    # model to update its state once the winner is known
    def undo(self):
        assert self._move_id != 0, 'Already at move 0.'
        self._move_id -= 1

        self._player_id, self._roll, idx, self._removed_opponent_stone = \
            self._history[self._move_id]

        self._state[:, :, 2] = self._roll
        self._undo_move(idx)

        self._player_stones = self._get_player_stone_indices()
        self._opponent_stones = self._get_opponent_stone_indices()

        self._options = self._get_options()

    # helpers for human game visualization
    @property
    def roll(self):
        return self._roll

    def has_stone_at_index(self, player_id, idx):
        # helper for plotting
        return idx in (self._player_stones if player_id == self._player_id
                       else self._opponent_stones)

    def get_move_idx(self, move_id=None):
        move_id = self._move_id if move_id is None else move_id
        return self._history[move_id][2]

    def print_roll_history(self, names=['Player 0', 'Player 1']):
        rolls0 = [d[1] for d in self._history if d[0] == 0]
        rolls1 = [d[1] for d in self._history if d[0] == 1]

        (r0, c0), m0 = np.unique(rolls0, return_counts=True), np.mean(rolls0)
        (r1, c1), m1 = np.unique(rolls1, return_counts=True), np.mean(rolls1)

        h1 = [str(f'{r}({c})') for r, c in zip(r0, c0)]
        h2 = [str(f'{r}({c})') for r, c in zip(r1, c1)]

        print('Roll History:')
        print(f"{names[0]}: {', '.join(h1)} (mean={m0:.2f})")
        print(f"{names[1]}: {', '.join(h2)} (mean={m1:.2f})")


if __name__ == '__main__':
    print('Hi')
    gm = GameManager(None)
    gm.start()
