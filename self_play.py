import logging
#from tqdm import tqdm
from tqdm_loggable.auto import tqdm

from mcts import mcts, Node, Game

logger = logging.getLogger(__name__)


def self_play(game: Game, n_rollout: int, cutoff: int, root: Node, desc="Self-play"):
    node = root
    reverse_q = False

    with tqdm(desc=desc) as pbar:
        while True:
            current_steps = pbar.n + 1

            if current_steps >= cutoff:
                break

            next_idx = mcts(game, n_rollout, node, reverse_q, cutoff)
            if next_idx is None:
                break

            # prune the other branches to save memory
            # we have recorded the statistics, and don't revisit the node
            # in this round a self-play.
            # but deleting the branch could make hurt the next round. A
            # better strategy can be delete only nodes deeper than 100 in
            # those branches.
            #for i in range(0, len(node.children)):
            #    if i == next_idx:
            #        continue
            #    prune(node.children[i], current_steps + 1, cutoff - current_steps - 1)

            node = node.children[next_idx]
            node.n_act += 1
            reverse_q = not reverse_q

            pbar.update()
            if pbar.n % 20 == 0:
                logger.info(f"\n{game.replay(node)}")

            #print(f"chosen {next_idx}")
            #for i, n in enumerate(root.children):
            #    print(f"root child {i:03}: nsa: {n.n_act}, q: {n.q}, n: {n.n_vis}")

    return node
