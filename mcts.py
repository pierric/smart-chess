import logging
import math
import random
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, List, Protocol, Tuple

import numpy as np
from tqdm import trange


logger = logging.getLogger(__name__)


EPSILON = 1e-4
CPUCT = 1.2

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

    def predict(self, board: BoardType) -> Tuple[bool, List[StepType], List[float], float]:
        """
        Given the board, predict the distribution of the next move and outcome
        """


def uct(sqrt_total_num_vis, prior, move_q, move_n_act, reverse_q):

    average_award = move_q / (move_n_act + EPSILON)
    if reverse_q:
        average_award = -average_award

    # plus EPSILON to ensure that exploration factor isn't zero
    # in case q and n_act are zero, the choice will fully based on the prior
    exploration = (sqrt_total_num_vis + EPSILON) / (1 + move_n_act) * CPUCT * prior
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

    while True:
        #sqrt_total_num_act = math.sqrt(sum(c.n_act for c in node.children))
        #
        # the more times the node is visited, the more likely to explore those less-visited children
        #
        _, moves, prior, outcome = game.predict(node, choose_max=False)

        if not node.children:
            # reaching a leaf node, either game is done, or it isn't finished, then
            # we expand the node (in the future if ever chosen the same node, will go
            # one level deeper). In both case, the predicted outcome is the result.

            for step in moves:
                node.children.append(Node(step, 0, 0, node, []))

            return (node, path_length, outcome)

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


#def simulate(game: Game, node: Node, cutoff: int, choose_max: bool) -> Tuple[Node, int, float]:
#    steps = 0
#
#    while True:
#
#        give_up, moves, distr, outcome = game.predict(node, choose_max=choose_max)
#
#        if steps >= cutoff or len(distr) == 0 or give_up:
#            return node, steps, outcome
#
#        if node.children:
#            assert len(node.children) == len(moves)
#        else:
#            for step in moves:
#                node.children.append(Node(step, 0, 0, node, []))
#
#        steps += 1
#        node = np.random.choice(node.children, p=distr)
#        node.n_act += 1


def backward(node: Node, reward: float, length: int):
    cnt = 0
    while node is not None and cnt < length:
        cnt += 1
        node.q += reward
        node = node.parent


def mcts(
    game: Game,
    n_rollout: int,
    root: Node,
    reverse_q: bool,
) -> List[Tuple[Node, float]]:

    ends = []

    for _ in trange(n_rollout, desc="Roll-out", leave=False):

        leaf, sel_length, reward = select(game, root, reverse_q)
        backward(leaf, reward, sel_length)
        ends.append((leaf, reward))

    return ends

    # NOTE use the expected reward seems to be very bad, because it will basically select the same action
    # for all the run of self-play with the same search tree.
    # return max_index([expected_reward(c, reverse_q) for c in root.children])


def prune(root: Node, max_depth: int = 60):
    if root is None or max_depth == 0:
        return None

    for node in root.children:
        prune(node, max_depth - 1)

