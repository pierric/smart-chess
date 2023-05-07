import logging
import random
from datetime import datetime

import chess
import numpy as np
import mcts
from self_play import self_play
from encode import encode_board_history, encode_action, encode_prob
from beton import write_beton
import torch
import torch._dynamo
from accelerate import Accelerator

logger = logging.getLogger(__name__)

#torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.verbose=True

N_ROLLOUT = 50
MOVES_CUTOFF = 80

class Chess(mcts.Game):


    def start(self):
        return mcts.Node(None, 0, 0, None, [])

    def replay(self, node, keep_history=True):
        steps = []

        while node is not None:
            if node.state is not None:
                steps.append(node.state)
            node = node.parent

        steps.reverse()
        board = chess.Board()
        for move in steps:
            board.push(move)

        if not keep_history:
            board.clear_stack()

        return board

    def get_history(self, board, n_look_back):
        board = board.copy(stack=True)
        history = []
        for i in range(n_look_back):
            history.append(board.copy(stack=False))
            try:
                board.pop()
            except IndexError:
                break
        history.reverse()
        return history

    def legal_moves(self, board):
        return list(board.legal_moves)

    def predict(self, board):
        raise NotImplementedError

    def judge(self, board):
        if board.is_checkmate():
            return 1 if board.turn == chess.WHITE else -1

        if board.is_insufficient_material() or board.is_stalemate():
            return 0

        raise ValueError("game not finished.")

    def give_up_policy(self, board):
        return 0 if board.can_claim_draw() else None


class ChessWithRandomPlayer(Chess):
    def predict(self, board):
        n_moves = len(list(board.legal_moves))

        if n_moves == 0:
            return [], self.judge(board)

        return [1/n_moves] * n_moves, 0


class ChessWithTwoPlayer(Chess):
    def __init__(self, player1, player2):
        super().__init__()
        self.player1 = player1
        self.player2 = player2

    def predict(self, boards):
        board = boards[-1]
        moves = list(board.legal_moves)

        if len(moves) == 0:
            return [], self.judge(board)

        board_enc = encode_board_history(boards, 8)

        if board.turn == chess.WHITE:
            distr, outcome = self.player1(board_enc)
        else:
            distr, outcome = self.player2(board_enc)
            outcome = -outcome

        act_enc = [encode_action(board.turn, m) for m in moves]
        act_prob = distr[act_enc]
        return act_prob / act_prob.sum(), outcome


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
        with torch.inference_mode():
            prob, outcome = self.model(inp)
            prob = torch.nn.functional.softmax(prob[0],dim=0).detach().cpu().numpy()
            outcome = outcome[0].detach().cpu().numpy().item()
        return prob, min(1, max(-1, outcome))


def dump_training_dataset(filename, game, node, outcome):
    dataset = []

    path = []

    while node is not None:
        path.append(node)
        node = node.parent

    path.reverse()

    outcome_table = {
        "white": 0,
        "black": 1,
        "draw": 2,
        "unknown": 3,
    }
    outcome_int = outcome_table[outcome]

    for i, _ in enumerate(path[:-1]):

        #start = max(0, i - 8)
        #boards = [game.replay(n, keep_history=False) for n in path[start:i+1]]
        board = game.replay(path[i], keep_history=True)
        boards = game.get_history(board, 8)
        board_enc = encode_board_history(boards, 8)

        node = path[i]
        board = game.replay(node, keep_history=False)
        act_enc = np.array([encode_action(board.turn, m) for m in board.legal_moves], dtype=int)
        act_prob = encode_prob([c.n_act for c in node.children])
        target = np.zeros(8 * 8 * 73, dtype=np.float32)
        np.put_along_axis(target, act_enc, act_prob, 0)

        dataset.append((board_enc, target, outcome_int))

    write_beton(filename, dataset)


MODEL_VER = "v0"

if __name__ == "__main__":
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
    accelerator.load_state(f"{MODEL_VER}/checkpoint")
    model.eval()

    player1 = NNPlayer(model)
    player2 = RandomPlayer()

    game = ChessWithTwoPlayer(player1, player2)
    init = game.start()

    for idx in range(5):
        end = self_play(game, N_ROLLOUT, MOVES_CUTOFF, init, desc=f"Self-play (Epoch {idx})")

        outcome = game.replay(end, keep_history=False).outcome(claim_draw=True)
        if outcome is None:
            winner = "unknown"
        elif outcome.winner is None:
            winner = "draw"
        elif outcome.winner == chess.WHITE:
            winner = "white"
        elif outcome.winner == chess.BLACK:
            winner = "black"

        logger.info(f"Outcome: {winner} Root Q: {init.q}")

        for i, n in enumerate(init.children):
            logger.info(f"child {i:03}: nsa: {n.n_act}, q: {n.q}")

        #mcts.prune(init, max_depth=60)

    time = datetime.utcnow().strftime("%Y-%m-%d-%X")
    dump_training_dataset(f"{time}.beton", game, end, winner)
