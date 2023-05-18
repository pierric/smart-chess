import logging
#from tqdm import tqdm
from tqdm_loggable.auto import tqdm

from mcts import mcts, Node, Game, max_index, expected_reward

logger = logging.getLogger(__name__)


def self_play(game: Game, n_rollout: int, cutoff: int, root: Node, desc="Self-play"):
    node = root
    reverse_q = False

    with tqdm(desc=desc, leave=False) as pbar:
        while True:
            current_steps = pbar.n + 1

            if current_steps >= cutoff:
                break

            mcts(game, n_rollout, node, reverse_q, cutoff)

            if not node.children:
                break

            next_idx = max_index([expected_reward(c, reverse_q) for c in node.children])

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
            #if pbar.n % 20 == 0:
            #    logger.info(f"\n{game.replay(node)}")

    return node
