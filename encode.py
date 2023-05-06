"""
copied from gym-chess https://github.com/iamlucaswolf/gym-chess
"""

from typing import Tuple
from unittest.mock import Mock

import chess
import numpy as np

import utils
import queenmoves
import knightmoves
import underpromotions


class BoardHistory:
    """Maintains a history of recent board positions, encoded as numpy arrays.
    New positions are added to the history via the push() method, which will
    store the position as a numpy array (using the encoding described in
    [Silver et al., 2017]). The history only retains the k most recent board
    positions; older positions are discarded when new ones are added. An array
    view of the history can be obtained via the view() function.

    Args:
        length: The number of most recent board positions to retain (corresponds
        to the 'k' parameter above).
    """

    def __init__(self, length: int) -> None:

        #: Ring buffer of recent board encodings; stored boards are always
        #: oriented towards the White player.
        self._buffer = np.zeros((length, 8, 8, 14), dtype=np.int32)


    def push(self, board: chess.Board) -> None:
        """Adds a new board to the history."""

        board_array = self.encode(board)

        # Overwrite oldest element in the buffer.
        self._buffer[-1] = board_array

        # Roll inserted element to the top (= most recent position); all older
        # elements are pushed towards the end of the buffer
        self._buffer = np.roll(self._buffer, 1, axis=0)


    def encode(self, board: chess.Board) -> np.array:
        #TODO optimize this function. It seems to be a hotspot
        """Converts a board to numpy array representation."""

        array = np.zeros((8, 8, 14), dtype=np.int32)

        for square, piece in board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color

            # The first six planes encode the pieces of the active player,
            # the following six those of the active player's opponent. Since
            # this class always stores boards oriented towards the white player,
            # White is considered to be the active player here.
            offset = 0 if color == chess.WHITE else 6

            # Chess enumerates piece types beginning with one, which we have
            # to account for
            idx = piece_type - 1

            array[rank, file, idx + offset] = 1

        # Repetition counters
        array[:, :, 12] = board.is_repetition(2)
        array[:, :, 13] = board.is_repetition(3)

        return array

    def view(self, orientation: bool = chess.WHITE) -> np.array:
        """Returns an array view of the board history.
        This method returns a (8, 8, k * 14) array view of the k most recently
        added positions. If less than k positions have been added since the
        last reset (or since the class was instantiated), missing positions are
        zeroed out.

        By default, positions are oriented towards the white player; setting the
        optional orientation parameter to 'chess.BLACK' will reorient the view
        towards the black player.

        Args:
            orientation: The player from which perspective the positions should
            be encoded.
        """

        # Copy buffer to not let reorientation affect the internal buffer
        array = self._buffer.copy()

        if orientation == chess.BLACK:
            for board_array in array:

                # Rotate all planes encoding the position by 180 degrees
                rotated = np.rot90(board_array[:, :, :12], k=2)

                # In the buffer, the first six planes encode white's pieces;
                # swap with the second six planes
                rotated = np.roll(rotated, axis=-1, shift=6)

                np.copyto(board_array[:, :, :12], rotated)

        # Concatenate k stacks of 14 planes to one stack of k * 14 planes
        array = np.concatenate(array, axis=-1)
        return array


    def reset(self) -> None:
        """Clears the history."""
        self._buffer[:] = 0



class MoveEncoding:

    def encode(self, turn, move: chess.Move) -> int:
        """Converts a `chess.Move` object to the corresponding action.
        This method converts a `chess.Move` instance to the corresponding
        integer action for the current board position.

        Args:
            action: The action to decode.
        Raises:
            ValueError: If `move` is not a valid move.
        """

        if turn == chess.BLACK:
            move = utils.rotate(move)

        # Successively try to encode the given move as a queen move, knight move
        # or underpromotion. If `move` is not of the associated move type, the
        # `encode` function in the resepctive helper modules will return None.

        action = queenmoves.encode(move)

        if action is None:
            action = knightmoves.encode(move)

        if action is None:
            action = underpromotions.encode(move)

        # If the move doesn't belong to any move type (i.e. every `encode`
        # functions returned None), it is considered to be invalid.

        if action is None:
            raise ValueError(f"{move} is not a valid move")

        #action = np.ravel_multi_index(
        #    multi_index=(index),
        #    dims=(8, 8, 73)
        #)

        return action


    def decode(self, turn, action: int) -> chess.Move:
        """Converts an action to the corresponding `chess.Move` object.

        This method converts an integer action to the corresponding `chess.Move`
        instance for the current board position.

        Args:
            action: The action to decode.
        Raises:
            ValueError: If `action` is not a valid action.
        """

        # Successively try to decode the given action as a queen move, knight
        # move, or underpromotion. If `index` does not reference the region
        # in the action array associated with the given move type, the `decode`
        # function in the resepctive helper module will return None.

        move = queenmoves.decode(action)
        is_queen_move = move is not None

        if not move:
            move = knightmoves.decode(action)

        if not move:
            move = underpromotions.decode(action)

        if not move:
            raise ValueError(f"{action} is not a valid action")

        if turn == chess.BLACK:
            move = utils.rotate(move)

        # Moving a pawn to the opponent's home rank with a queen move
        # is automatically assumed to be queen underpromotion. However,
        # since queenmoves has no reference to the board and can thus not
        # determine whether the moved piece is a pawn, we have to add this
        # information manually here
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == chess.WHITE) or
                (to_rank == 0 and turn == chess.BLACK)
            )

            return move, is_promoting_move

        return move, None


def encode_board_history(boards, length):
    benc = BoardHistory(length=length)

    for board in boards:
        benc.push(board)

    board = boards[-1]
    array = benc.view(orientation=board.turn)

    meta = np.zeros(
        shape=(8 ,8, 7),
        dtype=np.int32
    )

    meta[:, :, 0] = int(board.turn)

    meta[:, :, 1] = board.fullmove_number

    meta[:, :, 2] = board.has_kingside_castling_rights(board.turn)
    meta[:, :, 3] = board.has_queenside_castling_rights(board.turn)

    meta[:, :, 4] = board.has_kingside_castling_rights(not board.turn)
    meta[:, :, 5] = board.has_queenside_castling_rights(not board.turn)

    meta[:, :, 6] = board.halfmove_clock

    return np.concatenate([array, meta], axis=-1)


TEMPERATURE = 1

def encode_action(turn, move):
    return MoveEncoding().encode(turn, move)


def encode_prob(counts):
    counts = [x ** (1. / TEMPERATURE) for x in counts]
    counts_sum = float(sum(counts))
    return [x / counts_sum for x in counts]
