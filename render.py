from tempfile import mkdtemp
import os
from pathlib import Path
from subprocess import check_call

import matplotlib.cm
import ffcv
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder
import click
import numpy as np
import chess.svg

from encode import decode_board, MoveEncoding


@click.group()
def main():
    pass


@main.command()
@click.argument("records_file", default="records.txt")
def records(records_file):
    with open(records_file, "r") as rf:

        tmpdir = Path(mkdtemp())
        print(f"### Generating svgs in {tmpdir} ###")

        for idx, line in enumerate(rf):
            path = tmpdir / str(idx)
            path.mkdir()
            board = chess.Board()

            for s, m in enumerate(line.strip().split(",")):
                board.push(chess.Move.from_uci(m))
                img = chess.svg.board(board, orientation=board.turn, flipped=board.turn==chess.BLACK, size=350)
                (path / f"{s:02d}.svg").write_text(img)


@main.command()
@click.argument("beton_file")
def beton(beton_file):
    dl = ffcv.Loader(
        beton_file,
        batch_size=1,
        order=ffcv.loader.OrderOption.SEQUENTIAL,
        pipelines={
            "board": [NDArrayDecoder()],
            "move": [NDArrayDecoder()],
            "outcome": [IntDecoder()],
            "taken_action": [IntDecoder()],
        }
    )

    imgs = []
    outcome = None

    for item in dl:
        boards = item[0][0]
        actions = item[1][0]
        outcome = item[2][0].item()
        taken = item[3][0].item()

        board, turn, _ = decode_board(boards)

        moves = list(board.legal_moves)
        actions_enc = np.array([MoveEncoding.encode(turn, a) for a in moves])
        actions_prb = actions[actions_enc]
        actions_prb /= actions_prb.sum()

        # it's not always the move with lagest n_act that be chosen as the step
        # instead, the self-play calculates the average award. However, the accumulated
        # award isn't recorded in the beton file. So we cannot recover the move here.
        arrows = []
        #arrows = [chess.svg.Arrow(m.from_square, m.to_square, color="green") for m in moves]

        taken, _ = MoveEncoding.decode(turn, taken)
        arrows += [chess.svg.Arrow(taken.from_square, taken.to_square, color="red")]

        imgs.append(chess.svg.board(board, orientation=turn, arrows=arrows, flipped=turn==chess.BLACK, size=350))

    tmpdir = mkdtemp()
    print(f"### Generating svgs in {tmpdir}, outcome {outcome} ###")
    for idx, svg in enumerate(imgs):
        p = os.path.join(tmpdir, f"{idx:02d}.svg")
        with open(p, "w") as f:
            f.write(svg)


if __name__ == "__main__":
    main()
