import math
import torch
import collections
import numpy as np
from networkx import Graph, coloring
import sys
from matplotlib import pyplot as plt
from typing import *

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class Node(object):
  """MCTS node."""

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.reward = 0

  def expanded(self) -> bool:
    return bool(self.children)

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[int], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: int):
    self.history.append(action)

  def last_action(self) -> int:
    return self.history[-1]

  def action_space(self) -> Sequence[int]:
    return [i for i in range(self.action_space_size)]

  def to_play(self) -> int:
    return -1
  
  def __repr__(self) -> str:
    return "ActionHistory: " + str(self.history)


class Target(NamedTuple):
  correctness_value: float
  latency_value: float
  policy: Sequence[int]
  bootstrap_discount: float


class Sample(NamedTuple):
  observation: Dict[str, np.ndarray]
  bootstrap_observation: Dict[str, np.ndarray]
  target: Target
  
  
class TaskSpec(NamedTuple):
    nqubits: int
    budget: int
    board_dim: tuple[int, int]
    board_size: int
    gate_swap_penalty: float
    original_board: torch.FloatTensor
    atom_map: list[int]
    gate_action_idx: int
    num_tasks: int
    correct_reward: float
    correctness_reward_weight: float
    latency_reward_weight: float
    latency_quantile: float
    nvib_coeff: float
    feature_count: int
    num_actions: int
    threshold: int
    travel_distance_weight: float
    feature_type: str
    reward_type: str



class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class NetworkOutput(NamedTuple):
  value: float
  correctness_value_logits: torch.FloatTensor
  latency_value_logits: torch.FloatTensor
  policy_logits: list[float]

##########################

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_horizontal(self):
        return self.x == 0

    def is_vertical(self):
        return self.y == 0

    def is_left_to_right(self):
        return self.x > 0

    def is_bottom_to_top(self):
        return self.y > 0


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Point(self.x, self.y)

    def __eq__(self, p):
        if not isinstance(p, Point):
            return False
        if self.x == p.x and self.y == p.y:
            return True

    def __hash__(self):
        return hash((self.x, self.y))

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other: Vector):
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"({self.x},{self.y})"


class Move:
    """
    The transition from one point to another point.
    """

    def __init__(self, frm: Point, to: Point):
        self.frm = frm
        self.to = to

    def inverse(self):
        return Move(self.to, self.frm)

    def dist(self):
        return math.sqrt((self.frm.x - self.to.x) ** 2 + (self.frm.y - self.to.y) ** 2)

    def __eq__(self, __value):
        if not isinstance(__value, Move):
            return False
        return self.frm == __value.frm and self.to == __value.to

    def __hash__(self):
        return hash((self.frm, self.to))

    def __repr__(self):
        return f"{self.frm}->{self.to}"


class Board:
    def __init__(self, points: list[Point], atom_map: Dict[int, int]):
        """
        points:
        atom_map
        """
        self.points = points
        self.atom_map = atom_map

    def get_atom_position_map(self):
        result = {}
        for atom_id, point_id in self.atom_map.items():
            result[atom_id] = self.points[point_id]
        return result

    def get_atom_position(self, atom_id) -> Point:
        return self.points[self.atom_map[atom_id]]

    def get_atom_move(self, atom_id, grid_id) -> Move:
        if self.atom_map[atom_id] == grid_id:
            return None
        return Move(self.get_atom_position(atom_id), self.points[grid_id])

    def transform(self, a_id, grid_id, check_empty=True):
        if grid_id == self.atom_map[a_id]:
            return self
        if check_empty and grid_id not in set(self.empties()):
            raise AttributeError(f"{grid_id} is not empty.")
        a_map = self.atom_map.copy()
        a_map.pop(a_id)
        a_map[a_id] = grid_id
        return Board(self.points, a_map)

    def size(self):
        return len(self.points)

    def movable_mask(self, player):
        oks = set(self.empties())
        oks.add(self.atom_map[player])
        results = []
        for j in range(len(self.points)):
            if j in oks:
                results.append(0)
            else:
                results.append(1)
        return results

    def empties(self):
        """
        Return list of empty grids
        """
        results = []
        filled = set(self.atom_map.values())
        for j in range(len(self.points)):
            if j not in filled:
                results.append(j)
        return results

    def width(self):
        min_x = sys.maxsize
        max_x = -sys.maxsize
        for p in self.points:
            if min_x > p.x:
                min_x = p.x
            if max_x < p.x:
                max_x = p.x
        return max_x - min_x

    def height(self):
        min_y = sys.maxsize
        max_y = -sys.maxsize
        for p in self.points:
            if min_y > p.y:
                min_y = p.y
            if max_y < p.y:
                max_y = p.y
        return max_y - min_y

    def draw(self):
        plt.clf()
        fig, ax = plt.subplots()
        xs = []
        ys = []
        for p in self.points:
            xs.append(p.x)
            ys.append(p.y)
        plt.plot(xs, ys, linewidth=0, marker='.')

        for id, p_id in self.atom_map.items():
            point = self.points[p_id]
            plt.annotate(str(id), (point.x, point.y), bbox={"boxstyle": "circle", "color": "gray"})

        fig_size = self.fig_size()
        plt.xlim(fig_size[0], fig_size[1])
        plt.ylim(fig_size[2], fig_size[3])
        return fig, ax

    def fig_size(self):
        return [-self.width() * 0.2, self.width() * 1.2, -self.height() * 0.2, self.height() * 1.2]

    def vector(self):
        inverse_map = {}
        for a_id, p_id in self.atom_map.items():
            inverse_map[p_id] = a_id

        results = []
        for j, point in enumerate(self.points):
            if j not in inverse_map:
                results.append(-1)
            else:
                results.append(inverse_map[j])
        return results


class ParallelMove:
    """
    The class wraps multiple moves.
    """

    def __init__(self, moves: list[Move]):
        self.moves = moves

    def inverse(self):
        results = []
        for move in self.moves:
            results.append(move.inverse())
        return ParallelMove(results)

    def count(self):
        return len(self.moves)

    def __eq__(self, __value):
        if not isinstance(__value, ParallelMove):
            return False
        return self.moves == __value.moves

    def __repr__(self):
        return f"AOD:{str(self.moves)}"


class ParallelGrouping:
    cache = {}

    @staticmethod
    def to_groups(moves: list[Move]) -> list[ParallelMove]:
        """
        Converts a list of moves to a group of parallel executable moves.
        """
        graph = ParallelGrouping.to_graph(moves)
        move_to_label = coloring.greedy_color(graph, strategy="largest_first")
        group_to_move = {}
        for move, label in move_to_label.items():
            if label not in group_to_move:
                group_to_move[label] = []
            group_to_move[label].append(move)
        results = []
        for _, moves in group_to_move.items():
            results.append(ParallelMove(moves))
        return results

    @staticmethod
    def to_graph(moves: list[Move]) -> Graph:
        """
        Each node corresponds to each move.
        Non-parallel executable moves are connected.
        """
        graph = Graph()
        for m1 in moves:
            graph.add_node(m1)
            for m2 in moves:
                if not ParallelGrouping.is_parallel_executable(m1, m2):
                    graph.add_edge(m1, m2)
        return graph

    @staticmethod
    def is_parallel_executable(m1: Move, m2: Move):
        """
        We check if the relative positions are topologically the same.
        """
        v1 = m2.frm - m1.frm
        v2 = m2.to - m1.to

        hol1 = v1.is_horizontal()
        left_to_right1 = v1.is_left_to_right()

        hol2 = v2.is_horizontal()
        left_to_right2 = v2.is_left_to_right()

        horizontally_ok = False
        if hol1 and hol2:
            horizontally_ok = True
        elif hol1 or hol2:
            horizontally_ok = False
        elif left_to_right1 == left_to_right2:
            horizontally_ok = True
        if not horizontally_ok:
            return False

        ver1 = v1.is_vertical()
        ver2 = v2.is_vertical()
        if ver1 and ver2:
            return True
        elif ver1 or ver2:
            return False

        bottom_to_top1 = v1.is_bottom_to_top()
        bottom_to_top2 = v2.is_bottom_to_top()
        return bottom_to_top1 == bottom_to_top2




class Reward:
    def __init__(self, nvib_coeff):
        self.nvib_coeff = nvib_coeff

    def nvib_gain(self, twoq_gates, move):
        return self.nvib_coeff * move.dist() * twoq_gates + self.nvib_coeff

    def nvib_gain_vector(self, player, twoq_gates, original: Board):
        results = [-2] * original.size()
        results[original.atom_map[player]] = 0
        for grid_id in original.empties():
            move = original.get_atom_move(player, grid_id)
            if move is not None:
                gain = self.nvib_gain(twoq_gates, move)
            else:
                gain = 0
            results[grid_id] = gain
        return results

    @staticmethod
    def gain(board: Board, twoq_gates):
        moves = Reward.moves(board, twoq_gates)
        return len(moves) - len(ParallelGrouping.to_groups(moves))

    @staticmethod
    def move_gain(histories: list[Move]):
        return -len(ParallelGrouping.to_groups(histories))

    @staticmethod
    def gain_vector(player, original: Board, original_gain, pms: list[ParallelMove], get_move, mask):
        if mask:
            grid_ids = original.empties()
        else:
            grid_ids = [j for j in range(len(original.points))]
        results = [0] * original.size()
        for grid_id in grid_ids:
            move = get_move(grid_id)
            move_gain = 0 if Reward.increases_move_count_approximate(pms, move) else 1
            gain = move_gain - original_gain
            results[grid_id] = gain
        results[original.atom_map[player]] = 0
        return results

    @staticmethod
    def move_gain_vector(player, original: Board, histories: list[Move]):
        pms = ParallelGrouping.to_groups(histories)

        def get_move(grid_id):
            return original.get_atom_move(player, grid_id)

        return Reward.gain_vector(player, original, 1, pms, get_move, pms)

    @staticmethod
    def gate_gain_vector(player, original: Board, twoq_gates, mask=True):
        non_player_gates = []
        player_gate = None
        for q in twoq_gates:
            if q[0] != player and q[1] != player:
                non_player_gates.append(q)
            else:
                player_gate = q
        if player_gate is None:
            return [0] * original.size()
        moves = Reward.moves(original, non_player_gates)
        _, move = Reward.move(original, player_gate)
        pms = ParallelGrouping.to_groups(moves)
        original_gain = 0 if Reward.increases_move_count_approximate(pms, move) else 1

        def get_move(grid_id):
            board = original.transform(player, grid_id, check_empty=mask)
            _, move = Reward.move(board, player_gate)
            return move

        return Reward.gain_vector(player, original, original_gain, pms, get_move, mask)

    @staticmethod
    def parallel_moves(board: Board, twoq_gates):
        return ParallelGrouping.to_groups(Reward.moves(board, twoq_gates))

    @staticmethod
    def increases_move_count_approximate(original: list[ParallelMove], additional: Move):
        """
        Guess if the additional move increases the parallel count
        """
        for pm in original:
            parallel_executable = True
            pm: ParallelMove = pm
            for m in pm.moves:
                if not ParallelGrouping.is_parallel_executable(m, additional):
                    parallel_executable = False
                    break
            if parallel_executable:
                return False
        return True

    @staticmethod
    def increases_move_count(original: list[ParallelMove], additional: Move):
        moves = [additional]
        for pm in original:
            for m in pm.moves:
                moves.append(m)
        return len(ParallelGrouping.to_groups(moves)) - len(original)

    @staticmethod
    def moves(board: Board, twoq_gates):
        results = []
        for g in twoq_gates:
            _, move = Reward.move(board, gate=g)
            results.append(move)
        return results

    @staticmethod
    def move(board, gate):
        p_c: Point = board.get_atom_position(gate[0])
        p_t: Point = board.get_atom_position(gate[1])
        if p_c.x < p_t.x:
            p_1, p_2 = p_c, p_t
            g_1, g_2 = gate[0], gate[1]
        elif p_c.x < p_t.x:
            p_1, p_2 = p_t, p_c
            g_1, g_2 = gate[1], gate[0]
        else:
            if p_c.y < p_t.y:
                p_1, p_2 = p_c, p_t
                g_1, g_2 = gate[0], gate[1]
            else:
                p_1, p_2 = p_t, p_c
                g_1, g_2 = gate[1], gate[0]
        return (g_1, g_2), Move(p_1, p_2)


############################


from functools import partial
from torch import nn
from einops.layers.torch import Rearrange

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim)
    )
