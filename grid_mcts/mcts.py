import math
import torch
import numpy as np
from typing import Sequence
from .game import Game
from .env import NeutralAtomsEnv2
from .utils import Node, ActionHistory, MinMaxStats, NetworkOutput
from .config import NeutralAtomsConfig
from .network import Network
from .game import Game


def play_game(game: Game, config: NeutralAtomsConfig, network: Network) -> Game:
  """Plays an AlphaDev game.

  Each game is produced by starting at the initial empty program, then
  repeatedly executing a Monte Carlo Tree Search to generate moves until the end
  of the game is reached.

  Args:
    config: An instance of the AlphaDev configuration.
    network: Networks used for inference.

  Returns:
    The played game.
  """

  # print("Playing game")
  # print(game.tasks)
  # print('\n'.join([' '.join(f'{int(y):>3d}' if y >= 0 else '   ' for y in x) for x in (game.environment.board).tolist()]))
  while not game.terminal() and len(game.history) < config.max_moves:
    min_max_stats = MinMaxStats(config.known_bounds)

    # Initialisation of the root node and addition of exploration noise
    root = Node(0)
    current_observation = game.make_observation(-1)
    with torch.no_grad():
      network_output = network.inference(current_observation, aslist=True)
    _expand_node(
        root, game.to_play(), game.legal_actions(), network_output, reward=0
    )
    _backpropagate(
        [root],
        network_output.value,
        game.to_play(),
        config.discount,
        min_max_stats,
    )
    _add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using the environment.
    run_mcts(
        config,
        root,
        game.action_history(),
        network,
        min_max_stats,
        game.environment,
    )
    action = _select_action(config, len(game.history), root, network)
    if action == game.environment.gate_action:
      print('applying gate action...')
      print('actions history:', game.environment.actions)
      print('\n'.join([' '.join(f'{int(y):>3d}' if y >= 0 else '   ' for y in x) for x in (game.environment.board).tolist()]))
    # else:
    #   qubit, position = game.environment.action2qp(action)
    #   x, y = position // config.task_spec.board_dim[1], position % config.task_spec.board_dim[1]
    #   print(f'moving player {qubit} to position ({x}, {y})...')
    game.apply(action)
    game.store_search_statistics(root)
  print('game ended...')
  print('actions history:', game.environment.actions)
  print('\n'.join([' '.join(f'{int(y):>3d}' if y >= 0 else '   ' for y in x) for x in (game.environment.board).tolist()]))
  print('')
  return game


def run_mcts(
    config: NeutralAtomsConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    min_max_stats: MinMaxStats,
    env: NeutralAtomsEnv2,
):
  """Runs the Monte Carlo Tree Search algorithm.

  To decide on an action, we run N simulations, always starting at the root of
  the search tree and traversing the tree according to the UCB formula until we
  reach a leaf node.

  Args:
    config: AlphaDev configuration
    root: The root node of the MCTS tree from which we start the algorithm
    action_history: history of the actions taken so far.
    network: instances of the networks that will be used.
    min_max_stats: min-max statistics for the tree.
    env: an instance of the NeutralAtomsEnv2.
  """

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]
    sim_env = env.clone()

    while node.expanded():
      action, node = _select_child(config, node, min_max_stats)
      observation, reward = sim_env.step(action)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the environment to obtain the next
    # observation and reward given an action.
    with torch.no_grad():
      network_output = network.inference(observation, aslist=True)
    _expand_node(
        node, history.to_play(), sim_env.legal_actions(), network_output, reward
    ) # changed from history.action_space() --> sim_env.legal_actions()

    _backpropagate(
        search_path,
        network_output.value,
        history.to_play(),
        config.discount,
        min_max_stats,
    )


def _select_action(
    config: NeutralAtomsConfig, num_moves: int, node: Node, network: Network
):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      steps=network.training_steps()
  )
  # print('top three visit_counts:', sorted(visit_counts, key=lambda x: x[0], reverse=True)[:3])
  action = _softmax_sample(visit_counts, t, config.task_spec.num_actions)
  return action


def _select_child(
    config: NeutralAtomsConfig, node: Node, min_max_stats: MinMaxStats
):
  """Selects the child with the highest UCB score."""
  _, action, child = max(
      (_ucb_score(config, node, child, min_max_stats), action, child)
      for action, child in node.children.items()
  )
  return action, child


def _ucb_score(
    config: NeutralAtomsConfig,
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
) -> float:
  """Computes the UCB score based on its value + exploration based on prior."""
  pb_c = (
      math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
      + config.pb_c_init
  )
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = min_max_stats.normalize(
        child.reward + config.discount * child.value()
    )
  else:
    value_score = 0
  return prior_score + value_score


def _expand_node(
    node: Node,
    to_play: int,
    actions: Sequence[int],
    network_output: NetworkOutput,
    reward: float,
):
  """Expands the node using value, reward and policy predictions from the NN."""
  node.to_play = to_play
  node.reward = reward
  policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: int,
    discount: float,
    min_max_stats: MinMaxStats,
):
  """Propagates the evaluation all the way up the tree to the root."""
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


def _add_exploration_noise(config: NeutralAtomsConfig, node: Node):
  """Adds dirichlet noise to the prior of the root to encourage exploration."""
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def _softmax_sample(visit_counts, temperature, action_space_size):
  """Softmax sample function."""
  def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
  actions_prob = np.full(action_space_size,-np.inf)
  for count, action_idx in visit_counts:
    actions_prob[action_idx] = count
  actions_prob = softmax_stable(actions_prob/temperature)
  return np.random.choice(action_space_size, p=actions_prob)