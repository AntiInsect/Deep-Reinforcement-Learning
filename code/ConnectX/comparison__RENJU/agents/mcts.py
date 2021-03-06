from core import MCTS, RandomPolicy, PoolRAVEPolicy, TraditionalPolicy
from .agent import Agent


class MCTSAgent(Agent):
    """
    Agent Based on Monte Carlo Tree Search.
    Use "c_iterations" or "c_duration" as constraint.
    """
    def __init__(self, policy=None, **constraint):
        self.mcts = MCTS(policy=policy, **constraint)

    def get_action(self, state):
        self.mcts.sync_with_board(state)
        return self.mcts.get_action(state)

    def eval_state(self, state):
        self.mcts.sync_with_board(state)
        Q, pi = self.mcts.eval_state(state)
        self.mcts.step_forward()
        return Q, pi, self.mcts.root.position

    def reset(self):
        self.mcts.reset()

    def __repr__(self):
        return "MCTS Agent with {}".format(self.mcts.policy.__class__.__name__)


def RandomMCTSAgent(c_puct, c_rollouts=5, **constraint):
    return MCTSAgent(
        policy=RandomPolicy(c_puct, c_rollouts),
        **constraint
    )
def RAVEAgent(c_puct, c_bias, **constraint):
    return MCTSAgent(
        policy=PoolRAVEPolicy(c_puct, c_bias),
        **constraint
    )
def TraditionalAgent(c_puct, c_bias=0.0, use_rave=False, **constraint):
    return MCTSAgent(
        policy=TraditionalPolicy(c_puct, c_bias, use_rave),
        **constraint
    )


def main():
    from .utils import botzone_interface
    from config import MCTS_CONFIG
    botzone_interface(TraditionalAgent(**MCTS_CONFIG))


if __name__ == "__main__":
    main()
