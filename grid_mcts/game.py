from .env import NeutralAtomsEnv2
from .utils import Node, TaskSpec, ActionHistory, Target
from typing import Sequence


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(
      self, tasks, action_space_size: int, discount: float, task_spec: TaskSpec
  ):
    self.tasks = tasks
    self.task_spec = task_spec
    self.environment = NeutralAtomsEnv2(tasks, task_spec)
    self.history = []
    self.rewards = []
    self.latency_reward = 0
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    # For sorting, a game is terminal if we sort all sequences correctly or
    # we reached the end of the buffer.
    env = self.environment
    plan_cost = len(env.non_gate_actions) + self.task_spec.num_tasks
    assert plan_cost <= self.task_spec.budget
    if plan_cost == self.task_spec.budget:
        return 1
    if len(env.tasks) == env.tasks_done:
        return 1
    return 0

  def is_correct(self) -> bool:
    # Whether the current algorithm solves the game.
    return self.environment.is_correct()

  def legal_actions(self) -> Sequence[int]:
    # Game specific calculation of legal actions.
    return self.environment.legal_actions()

  def apply(self, action: int):
    _, reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    if self.terminal() and self.is_correct():
      self.latency_reward = self.environment.latency_reward()

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (index for index in range(self.action_space_size))
    self.child_visits.append(
        [
            root.children[a].visit_count / sum_visits
            if a in root.children
            else 0
            for a in action_space
        ]
    )
    self.root_values.append(root.value())

  def make_observation(self, state_index: int):
    if state_index == -1:
      return self.environment.observation()
    env = NeutralAtomsEnv2(self.tasks, self.task_spec)
    observation = env.observation()
    for action in self.history[:state_index]:
      observation, _ = env.step(action)
    return observation

  def make_target(
      # pylint: disable-next=unused-argument
      self, state_index: int, td_steps: int, to_play: int
  ) -> Target:
    """Creates the value target for training."""
    # The value target is the discounted sum of all rewards until N steps
    # into the future, to which we will add the discounted boostrapped future
    # value.
    bootstrap_index = state_index + td_steps
    value = 0
    for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
      value += reward * self.discount**i  # pytype: disable=unsupported-operands

    if bootstrap_index < len(self.root_values):
      bootstrap_discount = self.discount**td_steps
    else:
      bootstrap_discount = 0

    return Target(
        value,
        self.latency_reward,
        self.child_visits[state_index],
        bootstrap_discount,
    )

  def to_play(self) -> int:
    return -1

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)
  
  def game_total_execs(self):
    return sum(self.environment.num_gate_executions(self.environment.tasks, self.history))

