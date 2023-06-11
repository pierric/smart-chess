import logging

from tqdm_loggable.auto import tqdm
import numpy as np

from mcts import mcts, Node, Game, max_index

logger = logging.getLogger(__name__)


def self_play(
    game: Game,
    n_rollout: int,
    cutoff: int,
    root: Node,
    temp:int = 1,
    desc="Play",
):
    node = root
    reverse_q = False

    # z indicates how many times model gives a bad prediction
    z = 0

    with tqdm(desc=desc, leave=False) as pbar:
        while True:
            current_steps = pbar.n + 1

            if current_steps >= cutoff:
                break

            _, z0 = mcts(game, n_rollout, node, reverse_q)
            z += z0
            pbar.set_postfix({"z": z})

            #if node.parent is None:
            #    root_children_status = ""
            #    for i, n in enumerate(node.children):
            #        root_children_status += f"child {i:03}: nsa: {n.n_act}, q: {n.q}\n"
            #    logger.info(root_children_status)

            if not node.children:
                break

            if temp == 0:
                next_idx = max_index([c.n_act for c in node.children])

            else:
                counts = np.array([c.n_act ** (1. / temp) for c in node.children])
                pi = counts / counts.sum()
                next_idx = np.random.choice(len(pi), p=pi)

            # prune the other branches to save memory
            # we have recorded the statistics, and don't revisit the node
            # in this round a self-play.
            # but deleting the branch could make hurt the next round. A
            # better strategy can be delete only nodes deeper than 100 in
            # those branches.
            for i in range(0, len(node.children)):
                if i == next_idx:
                    continue
                #prune(node.children[i], current_steps + 1, cutoff - current_steps - 1)
                node.children[i].children = []

            node = node.children[next_idx]
            #node.n_act += 1
            reverse_q = not reverse_q

            pbar.update()
            #if pbar.n % 20 == 0:
            #    logger.info(f"\n{game.replay(node)}")

    return node
