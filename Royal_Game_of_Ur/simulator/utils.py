"""
Helper functions used by GameManager and in the actual games.
"""

def roll_dice(rng):
    """ Rolls the dice in the same way as in the board game. """
    roll = rng.random()

    if roll < 0.065:  # 6.5%
        return 0

    if roll < 0.3125:  # 25%
        return 1

    if roll < 0.6875:  # 37.5%
        return 2

    if roll < 0.9375:  # 25%
        return 3

    return 4


def convert_to_rc(idx):
    """ Converts from the 1d board index to row/column indices. """
    return idx // 8, idx % 8


def convert_to_idx(r, c):
    """ Converts from the 2d row/column indices to the 1d board index. """
    return int(8*r+c)


def get_next_idx(player_id, idx, roll):
    """
    Given board position 'idx' and dice roll 'roll', find the next board index.
    """
    r, c = convert_to_rc(idx)

    if r == 2 * player_id:  # in protected row
        c -= roll

        if c < 0:
            r, c = 1, -c-1

        return convert_to_idx(r, c)

    # in middle row
    c += roll

    if c >= 8:
        r, c = 2*player_id, 15-c

    return convert_to_idx(r, c)


def get_prev_idx(player_id, idx, roll):
    """
    Given board position 'idx' and (previous) dice roll 'roll',
    find the previous board index.
    """
    r, c = convert_to_rc(idx)

    if r == 2 * player_id:  # in protected row
        c += roll

        if c >= 8:
            r, c = 1, 15-c

        return convert_to_idx(r, c)

    # in middle row
    c -= roll

    if c < 0:
        r, c = 2*player_id, -c-1

    return convert_to_idx(r, c)


