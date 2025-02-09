from typing import List, Tuple

from mcts import MCTS, MCTSConfig, Node


class SimpleGridWorld:
    """A simple grid world environment for testing MCTS."""

    def __init__(self, size: int = 5, target: Tuple[int, int] = (4, 4)):
        self.size = size
        self.target = target
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    def get_valid_actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid actions from current state."""
        valid = []
        for action in self.actions:
            new_x = state[0] + action[0]
            new_y = state[1] + action[1]
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                valid.append(action)
        return valid

    def step(
        self, state: Tuple[int, int], action: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], float, bool]:
        """Take a step in the environment."""
        new_x = state[0] + action[0]
        new_y = state[1] + action[1]
        new_state = (new_x, new_y)

        # Check if reached target
        if new_state == self.target:
            return new_state, 1.0, True

        # Small negative reward for each step
        return new_state, -0.1, False


def test_node_update():
    """Test node statistics update."""
    node = Node(state=(0, 0))
    node.update(1.0)
    assert node.visit_count == 1
    assert node.total_reward == 1.0
    assert node.mean_reward == 1.0

    node.update(0.5)
    assert node.visit_count == 2
    assert node.total_reward == 1.5
    assert abs(node.mean_reward - 0.75) < 1e-6


def test_ucb1_basic():
    """Test basic UCB1 behavior."""
    parent = Node(state=(0, 0))

    # Create two children with different statistics
    child1 = parent.add_child(state=(1, 0), action=(1, 0))
    child1.visit_count = 10
    child1.mean_reward = 0.5

    child2 = parent.add_child(state=(0, 1), action=(0, 1))
    child2.visit_count = 5
    child2.mean_reward = 0.4

    # Less visited node should have higher UCB1 score due to exploration
    assert child2.ucb1_score(1.414) > child1.ucb1_score(1.414)


def test_selection_exploration():
    """Test that selection balances exploration and exploitation."""
    mcts = MCTS(MCTSConfig(exploration_constant=2.0))  # Higher exploration
    parent = Node(state=(0, 0))

    # Create children with very different visit counts
    child1 = parent.add_child(state=(1, 0), action=(1, 0))
    child1.visit_count = 50
    child1.mean_reward = 0.6

    child2 = parent.add_child(state=(0, 1), action=(0, 1))
    child2.visit_count = 5
    child2.mean_reward = 0.4

    # Run selection multiple times
    selections = {child1: 0, child2: 0}
    for _ in range(100):
        selected = mcts.select_child(parent)
        selections[selected] += 1

    # Both should be selected, with less visited node getting some chances
    assert selections[child1] > 0
    assert selections[child2] > 0
    # But better node should be selected more often
    assert selections[child1] > selections[child2]


def test_expansion():
    """Test node expansion."""
    env = SimpleGridWorld(size=2)  # Small grid for simple testing
    mcts = MCTS(MCTSConfig())
    root = Node(state=(0, 0))
    root.untried_actions = env.get_valid_actions(root.state)

    # Should expand to a new valid state
    child = mcts.expand(root, env)
    assert child in root.children
    assert child.state[0] >= 0 and child.state[0] < 2
    assert child.state[1] >= 0 and child.state[1] < 2


def test_simulation_path_to_goal():
    """Test that simulation can find path to goal."""
    env = SimpleGridWorld(size=3, target=(2, 2))
    mcts = MCTS(MCTSConfig(max_depth=10))

    # Run multiple simulations from start state
    rewards = []
    for _ in range(50):
        reward = mcts.simulate((0, 0), env)
        rewards.append(reward)

    # At least some simulations should reach goal
    assert any(r > 0 for r in rewards), "No simulation reached the goal"
    # Average reward should be reasonable
    avg_reward = sum(rewards) / len(rewards)
    assert avg_reward > -1.0, "Average reward too low"


def test_full_search_optimal_action():
    """Test that MCTS finds optimal action in simple scenario."""
    # Create a 2x2 grid with target at (1,1)
    env = SimpleGridWorld(size=2, target=(1, 1))
    config = MCTSConfig(num_simulations=100)
    mcts = MCTS(config)

    # Start at (0,0) - optimal actions are right or down
    start_state = (0, 0)
    action, value = mcts.search(start_state, env)

    # Action should move towards goal
    assert action in [(1, 0), (0, 1)], "Did not choose optimal action"
    # Value should be positive (path to goal found)
    assert value > 0, "Did not find positive value path"


def test_terminal_state():
    """Test handling of terminal states."""
    env = SimpleGridWorld(size=2, target=(1, 1))
    mcts = MCTS(MCTSConfig())

    # Test at goal state
    terminal_state = (1, 1)
    action, value = mcts.search(terminal_state, env)
    assert action is None and value == 0.0

    # Test simulation from terminal state
    reward = mcts.simulate(terminal_state, env)
    assert reward == 0.0
