from typing import List, Tuple

from .mcts import MCTS, MCTSConfig, Node


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

    # Test single update
    node.update(1.0)
    assert node.visit_count == 1
    assert node.total_reward == 1.0
    assert node.mean_reward == 1.0

    # Test multiple updates
    node.update(0.5)
    assert node.visit_count == 2
    assert node.total_reward == 1.5
    assert abs(node.mean_reward - 0.75) < 1e-6


def test_ucb1_score():
    """Test UCB1 score calculation."""
    parent = Node(state=(0, 0))
    parent.visit_count = 100

    child = Node(state=(1, 0), parent=parent)

    # Test unvisited node
    assert child.ucb1_score(1.414) == float('inf')

    # Test after some visits
    child.visit_count = 20
    child.mean_reward = 0.5
    score = child.ucb1_score(1.414)

    # UCB1 should be greater than just the mean reward
    assert score > child.mean_reward

    # Test with different exploration constants
    score1 = child.ucb1_score(1.0)
    score2 = child.ucb1_score(2.0)
    assert score2 > score1  # Higher exploration constant should give higher score


def test_select_child():
    """Test child selection using UCB1."""
    mcts = MCTS(MCTSConfig())
    parent = Node(state=(0, 0))

    # Add children with different statistics
    child1 = parent.add_child(state=(1, 0), action=(1, 0))
    child1.visit_count = 10
    child1.mean_reward = 0.6

    child2 = parent.add_child(state=(0, 1), action=(0, 1))
    child2.visit_count = 15
    child2.mean_reward = 0.4

    # Test selection
    selected_counts = {child1: 0, child2: 0}
    for _ in range(100):
        selected = mcts.select_child(parent)
        selected_counts[selected] += 1

    # Both children should be selected due to exploration
    assert selected_counts[child1] > 0
    assert selected_counts[child2] > 0
    # Child1 should be selected more often due to higher mean reward
    assert selected_counts[child1] > selected_counts[child2]


def test_expand():
    """Test node expansion."""
    env = SimpleGridWorld(size=3)
    mcts = MCTS(MCTSConfig())

    # Create root node
    root = Node(state=(0, 0))
    root.untried_actions = env.get_valid_actions(root.state)

    # Test expansion
    initial_untried = len(root.untried_actions)
    child = mcts.expand(root, env)

    assert len(root.untried_actions) == initial_untried - 1
    assert child in root.children
    assert child.parent == root
    assert child.state != root.state


def test_simulate():
    """Test simulation from a state."""
    env = SimpleGridWorld(size=3, target=(2, 2))
    mcts = MCTS(MCTSConfig(max_depth=10))

    # Run multiple simulations
    rewards = []
    for _ in range(10):
        reward = mcts.simulate((0, 0), env)
        rewards.append(reward)

    # Some simulations should reach the goal
    assert any(r > 0 for r in rewards)
    # Some simulations should hit max depth
    assert any(r < 0 for r in rewards)


def test_backpropagate():
    """Test reward backpropagation."""
    mcts = MCTS(MCTSConfig(discount_factor=0.9))

    # Create a simple tree
    root = Node(state=(0, 0))
    child = root.add_child(state=(1, 0), action=(1, 0))
    leaf = child.add_child(state=(2, 0), action=(1, 0))

    # Test backpropagation
    mcts.backpropagate(leaf, 1.0)

    # Check if rewards were properly discounted and propagated
    assert leaf.visit_count == 1
    assert leaf.total_reward == 1.0

    assert child.visit_count == 1
    assert abs(child.total_reward - 0.9) < 1e-6

    assert root.visit_count == 1
    assert abs(root.total_reward - 0.81) < 1e-6


def test_full_search():
    """Test the complete MCTS algorithm."""
    env = SimpleGridWorld(size=3, target=(2, 2))
    config = MCTSConfig(
        num_simulations=1000,
        max_depth=10,
        exploration_constant=1.414,
        discount_factor=0.9,
    )
    mcts = MCTS(config)

    # Run search from initial state
    action, value = mcts.search((0, 0), env)

    # Should select an action that moves towards the goal
    assert action in [(1, 0), (0, 1)]
    # Value should be positive (indicating path to goal was found)
    assert value > 0


def test_terminal_state_handling():
    """Test handling of terminal states."""
    env = SimpleGridWorld(size=2, target=(1, 1))
    mcts = MCTS(MCTSConfig(num_simulations=100))

    # Create terminal node
    terminal_node = Node(state=(1, 1), is_terminal=True)

    # Should not expand terminal nodes
    assert mcts.expand(terminal_node, env) == terminal_node

    # Simulation from terminal state should return 0
    assert mcts.simulate((1, 1), env) == 0

    # Search should handle terminal root state
    action, value = mcts.search((1, 1), env)
    assert value == 0
