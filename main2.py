import logging
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np
import mcts
from self_play import self_play
from encode import encode_boards, encode_action, encode_prob, BoardHistory
from beton import write_beton
import torch
import torch._dynamo
from accelerate import Accelerator
import tqdm
import click

logger = logging.getLogger(__name__)

#torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.verbose=True

@dataclass
class Step:
    move: Optional[chess.Move]
    encoded_board: Optional[np.ndarray]


class Chess(mcts.Game[Step, chess.Board]):

    def start(self):
        return mcts.Node(Step(None, None), 0, 0, None, [])

    def replay(self, node, keep_history=True):
        steps = []

        while node is not None:
            if node.state.move is not None:
                steps.append(node.state.move)
            node = node.parent

        steps.reverse()
        board = chess.Board()
        for move in steps:
            board.push(move)

        if not keep_history:
            board.clear_stack()

        return board

    def predict(self, node):
        raise NotImplementedError

    def judge(self, board):
        if board.is_checkmate():
            # IMPORTANT turn is black means white wins
            return 1 if board.turn == chess.BLACK else -1

        if board.is_insufficient_material() or board.is_stalemate():
            return 0

        raise ValueError("game not finished.")


class ChessWithTwoPlayer(Chess):
    def __init__(self, player1, player2):
        super().__init__()
        self.player1 = player1
        self.player2 = player2

    def predict(self, node):
        path = []
        cnt = 0

        tmp_node = node
        while tmp_node is not None and cnt < 8:
            path.append(tmp_node )
            tmp_node = tmp_node.parent
            cnt += 1

        path.reverse()

        board = self.replay(path[0], keep_history=False)

        # encode and cache the boards along the path
        for i, n in enumerate(path):
            if i > 0:
                board.push(n.state.move)

            if n.state.encoded_board is None:
                n.state.encoded_board = BoardHistory.encode(board)

        # now the board should be the last one
        #assert board.board_fen() == self.replay(node, keep_history=False).board_fen()

        moves = list(board.legal_moves)

        if len(moves) == 0:
            return False, [], [], self.judge(board)

        # give up playing
        if board.can_claim_draw():
            return True, [], [], 0

        board_enc = encode_boards([n.state.encoded_board for n in path], 8, board)

        if board.turn == chess.WHITE:
            distr, outcome = self.player1(board_enc)
        else:
            distr, outcome = self.player2(board_enc)
            outcome = -outcome

        act_enc = [encode_action(board.turn, m) for m in moves]
        act_prob = distr[act_enc]
        return False, [Step(m, None) for m in moves], act_prob / act_prob.sum(), outcome


class RandomPlayer:
    def __call__(self, board_enc):
        prob = np.ones(4672, dtype="float32")
        prob /= prob.size
        return prob, 0


class NNPlayer:
    def __init__(self, model):
        self.model = model

    def __call__(self, board_enc):
        inp = torch.tensor(board_enc).float().unsqueeze(0).cuda()
        inp = inp.permute(0, 3, 1, 2)
        with torch.inference_mode():
            prob, outcome = self.model(inp)
            prob = prob[0]
            outcome = outcome[0]
            prob = prob.detach().cpu().numpy()
            outcome = (outcome + outcome.sign()* 0.5).trunc().detach().cpu().numpy().item()
        return prob, outcome


def dump_training_dataset(filename, game, node, outcome):
    dataset = []

    path = []

    while node is not None:
        path.append(node)
        node = node.parent

    path.reverse()

    outcome_table = {
        "white": 1,
        "black": -1,
        "draw": 0,
        "unknown": 0,
    }
    outcome_int = outcome_table[outcome]

    for i, node in enumerate(path[:-1]):

        board = game.replay(node, keep_history=False)
        start = max(0, i - 8)
        boards = [n.state.encoded_board for n in path[start:i+1]]
        board_enc = encode_boards(boards, 8, board)

        act_enc = np.array([encode_action(board.turn, m) for m in board.legal_moves], dtype=int)
        act_prob = encode_prob([c.n_act for c in node.children])
        target = np.zeros(8 * 8 * 73, dtype=np.float32)
        np.put_along_axis(target, act_enc, act_prob, 0)

        # save the taken move so that I can render the full play from the beton file
        taken_act = encode_action(board.turn, path[i+1].state.move)

        dataset.append((board_enc, target, outcome_int, taken_act))

    write_beton(filename, dataset)


@click.command()
@click.option("--n-rollout", default=400)
@click.option("--moves-cutoff", default=60)
@click.option("--n-epochs", default=20)
@click.option("--model-ver")
@click.option("--model-prefix", default="v")
@click.option("--save-all", default=False)
def main(n_epochs, n_rollout, moves_cutoff, model_ver, model_prefix, save_all):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[logging.StreamHandler()]
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        dynamo_backend="inductor",
    )

    import train
    model = train.ChessModel()
    model = accelerator.prepare(model)
    if model_ver is not None:
        accelerator.load_state(f"{model_prefix}{model_ver}/checkpoint")
    model.eval()

    player1 = NNPlayer(model)
    #player2 = RandomPlayer()

    game = ChessWithTwoPlayer(player1, player1)

    postfix = {"w": 0, "b": 0, "d": 0, "u": 0}
    pbar = tqdm.tqdm(total=n_epochs)

    for idx in range(n_epochs):
        init = game.start()
        end_node = self_play(game, n_rollout, moves_cutoff, init, desc=f"Self-play (Epoch {idx})")

        outcome = game.replay(end_node, keep_history=False).outcome(claim_draw=True)
        if outcome is None:
            winner = "unknown"
            postfix["u"] += 1
        elif outcome.winner is None:
            winner = "draw"
            postfix["d"] += 1
        elif outcome.winner == chess.WHITE:
            winner = "white"
            postfix["w"] += 1
        elif outcome.winner == chess.BLACK:
            winner = "black"
            postfix["b"] += 1

        if idx % 2 == 0:
            logger.info(f"Outcome: {winner} Root Q: {init.q}")

            root_children_status = ""
            for i, n in enumerate(init.children):
                root_children_status += f"child {i:03}: nsa: {n.n_act}, q: {n.q}\n"
            logger.info(root_children_status)

        pbar.set_postfix(postfix)
        pbar.update()

        save_it = save_all or model_ver is None or winner in ["white", "black", "draw"]
        if save_it:
            time = datetime.utcnow().strftime("%Y-%m-%d-%X")
            dump_training_dataset(f"{time}.beton", game, end_node, winner)


if __name__ == "__main__":
    main()





