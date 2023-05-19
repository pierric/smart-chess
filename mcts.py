import logging
import math
import random
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, List, Protocol, Tuple

import numpy as np
from tqdm import trange


logger = logging.getLogger(__name__)


EPSILON = 1e-4
CPUCT = 2.0

BoardType = TypeVar("BoardType")
StepType = TypeVar("StepType")


class Node(Generic[StepType]):
    state: Optional[StepType]
    q: float
    n_act: int
    parent: Optional["Node[StepType]"]
    children: List["Node[StepType]"]

    def __init__(self, state, q, n_act, parent, children):
        self.state = state
        self.q = q
        self.n_act = n_act
        self.parent = parent
        self.children = children


class Game(Protocol, Generic[StepType, BoardType]):

    def replay(self, node: Node[StepType], keep_history: bool = True) -> BoardType:
        ...

    def start(self) -> Node[StepType]:
        ...

    def legal_moves(self, board: BoardType) -> List[StepType]:
        ...

    def predict(self, board: BoardType) -> Tuple[List[float], float]:
        """
        Given the board, predict the distribution of the next move and outcome
        """

    def judge(self, board: BoardType) -> float:
        ...

    def give_up_policy(self, board: BoardType) -> Optional[float]:
        ...


def uct(sqrt_total_num_vis, prior, move_q, move_n_act, reverse_q):

    average_award = move_q / (move_n_act + EPSILON)
    if reverse_q:
        average_award = -average_award

    exploration = sqrt_total_num_vis / (1 + move_n_act) * CPUCT * prior
    return average_award + exploration


def expected_reward(node: Node, reverse_q) -> float:
    v = node.q / (node.n_act + EPSILON)
    return v if not reverse_q else -v


def max_index(values: List[float]) -> int:
    values = np.array(values)
    best = np.flatnonzero(values == np.max(values))
    return random.choice(best)


def select(game: Game, node: Node, reverse_q: bool) -> Tuple[Node, int]:
    """
    Descend in the tree until some leaf, exploiting the knowledge to choose
    the best child each time.
    """
    path_length = 1

    board = game.replay(node)
    boards = game.get_history(board, 8)

    while node.children:
        #sqrt_total_num_act = math.sqrt(sum(c.n_act for c in node.children))
        #
        # the more times the node is visited, the more likely to explore those less-visited children
        #
        prior, _ = game.predict(boards)
        # adding the Dir(0.03) noise
        # https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
        prior = prior * 0.75 + np.random.dirichlet([0.03]*len(prior)) * 0.25
        sqrt_total_num_vis = math.sqrt(sum(c.n_act for c in node.children))
        uct_children = [
            uct(sqrt_total_num_vis, p, c.q, c.n_act, reverse_q) for p, c in zip(prior, node.children)
        ]
        #if node.parent is None:
        #    logger.info(" ".join(map(lambda v: f"{v:0.1}", uct_children)))
        node = node.children[max_index(uct_children)]
        node.n_act += 1
        path_length += 1
        board.push(node.state)
        boards.append(board)
        boards = boards[-8:]

    return (node, path_length)


def simulate(game: Game, node: Node, cutoff: int) -> Tuple[Node, int, float]:
    steps = 0

    board = game.replay(node)
    boards = game.get_history(board, 8)

    while True:

        distr, outcome = game.predict(boards)

        if steps >= cutoff or len(distr) == 0:
            return node, steps, outcome

        give_up_reward = game.give_up_policy(board)
        if give_up_reward is not None:
            return node, steps, give_up_reward

        assert len(node.children) == 0
        for move in game.legal_moves(board):
            node.children.append(Node(move, 0, 0, node, []))

        steps += 1
        node = np.random.choice(node.children, p=distr)
        node.n_act += 1
        board.push(node.state)
        boards.append(board)
        boards = boards[-8:]


def backward(node: Node, reward: float, length: int):
    cnt = 0
    while node is not None and cnt < length:
        cnt += 1
        node.q += reward
        node = node.parent


def mcts(game: Game, n_rollout: int, root: Node, reverse_q: bool, cutoff: int) -> List[Tuple[Node, float]]:
    ends = []

    for _ in trange(n_rollout, desc="Roll-out", leave=False):
        leaf, sel_length = select(game, root, reverse_q)
        last, sim_length, reward = simulate(game, leaf, cutoff - sel_length)
        backward(last, reward, sel_length + sim_length)
        ends.append((last, reward))

    return ends

    # NOTE use the expected reward seems to be very bad, because it will basically select the same action
    # for all the run of self-play with the same search tree.
    # return max_index([expected_reward(c, reverse_q) for c in root.children])


def prune(root: Node, max_depth: int = 60):
    if root is None or max_depth == 0:
        return None

    for node in root.children:
        prune(node, max_depth - 1)

