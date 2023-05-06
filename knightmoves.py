"""
Helper module to encode/decode knight moves.

copied from gym-chess https://github.com/iamlucaswolf/gym-chess

"""

import chess
import numpy as np
from numba import njit
from numba.core import types as nbtypes
from numba.typed import Dict as TypedDict

import utils

from typing import Optional

#: Number of possible knight moves
_NUM_TYPES: int = 8

#: Starting point of knight moves in last dimension of 8 x 8 x 73 action array.
_TYPE_OFFSET: int = 56

#: Set of possible directions for a knight move, encoded as 
#: (delta rank, delta square).
_DIRECTIONS = utils.IndexedTuple(
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
)

_DIRECTIONS_INDICES = _DIRECTIONS.indices_numba(nbtypes.Tuple((nbtypes.int64, nbtypes.int64)))


@njit
def _encode(
    from_rank: int,
    from_file: int,
    to_rank: int,
    to_file: int,
    directions_indices: nbtypes.DictType(nbtypes.int64, nbtypes.int64),
) -> Optional[int]:
    delta = (to_rank - from_rank, to_file - from_file)
    is_knight_move = delta in directions_indices

    if not is_knight_move:
        return None

    knight_move_type = directions_indices[delta]
    move_type = _TYPE_OFFSET + knight_move_type

    return from_rank * 8 * 73 + from_file * 73 + move_type
    #action = np.ravel_multi_index(
    #    multi_index=((from_rank, from_file, move_type)),
    #    dims=(8, 8, 73)
    #)

    #return action

def encode(move: chess.Move) -> Optional[int]:
    """Encodes the given move as a knight move, if possible.

    Returns:
        The corresponding action, if the given move represents a knight move; 
        otherwise None.

    """
    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    return _encode(from_rank, from_file, to_rank, to_file, _DIRECTIONS_INDICES)


def decode(action: int) -> Optional[chess.Move]:
    """Decodes the given action as a knight move, if possible.

    Returns:
        The corresponding move, if the given action represents a knight move; 
        otherwise None.

    """

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_knight_move:
        return None

    knight_move_type = move_type - _TYPE_OFFSET

    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move
