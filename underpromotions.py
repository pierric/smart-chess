"""
Helper module to encode/decode underpromotions.

copied from gym-chess https://github.com/iamlucaswolf/gym-chess
"""

import chess
import numpy as np
from numba import njit
from numba.core import types as nbtypes

import utils

from typing import Optional

#: Number of possible underpromotions 
_NUM_TYPES: int = 9 # = 3 directions * 3 piece types (see below)

#: Starting point of underpromotions in last dimension of 8 x 8 x 73 action 
#: array.
_TYPE_OFFSET: int = 64

#: Set of possibel directions for an underpromotion, encoded as file delta.
_DIRECTIONS = utils.IndexedTuple(
    -1,
     0,
    +1,
)

_DIRECTIONS_INDICES = _DIRECTIONS.indices_numba(nbtypes.int64)

#: Set of possibel piece types for an underpromotion (promoting to a queen
#: is implicitly encoded by the corresponding queen move).
_PROMOTIONS = utils.IndexedTuple(
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
)

_PROMOTIONS_INDICES = _PROMOTIONS.indices_numba(nbtypes.int64)


@njit
def _encode(
    from_rank: int,
    from_file: int,
    to_rank: int,
    to_file: int,
    promotion: int,
    directions_indices: nbtypes.DictType(nbtypes.int64, nbtypes.int64),
    promotions_indices: nbtypes.DictType(nbtypes.int64, nbtypes.int64),
) -> Optional[int]:

    is_underpromotion = (
        promotion in promotions_indices
        and from_rank == 6
        and to_rank == 7
    )

    if not is_underpromotion:
        return None

    delta_file = to_file - from_file

    direction_idx = directions_indices[delta_file]
    promotion_idx = promotions_indices[promotion]

    #underpromotion_type = np.ravel_multi_index(
    #    multi_index=([direction_idx, promotion_idx]),
    #    dims=(3,3)
    #)
    underpromotion_type = direction_idx * 3 + promotion_idx

    move_type = _TYPE_OFFSET + underpromotion_type

    #action = np.ravel_multi_index(
    #    multi_index=((from_rank, from_file, move_type)),
    #    dims=(8, 8, 73)
    #)

    #return action
    return from_rank * 8 * 73 + from_file * 73 + move_type


def encode(move):
    """Encodes the given move as an underpromotion, if possible.

    Returns:
        The corresponding action, if the given move represents an 
        underpromotion; otherwise None.

    """
    from_rank, from_file, to_rank, to_file = utils.unpack(move)
    return _encode(
        from_rank,
        from_file,
        to_rank,
        to_file,
        move.promotion,
        _DIRECTIONS_INDICES,
        _PROMOTIONS_INDICES,
    )



def decode(action):
    """Decodes the given action as an underpromotion, if possible.

    Returns:
        The corresponding move, if the given action represents an 
        underpromotion; otherwise None.

    """

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_underpromotion:
        return None

    underpromotion_type = move_type - _TYPE_OFFSET

    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    to_rank = from_rank + 1
    to_file = from_file + direction

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move
