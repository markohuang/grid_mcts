import torch
from itertools import chain, takewhile
from functools import cached_property
from .utils import Point, Move, ParallelGrouping, Board, Reward
from torch.distributions import Categorical

def entropy(x):
    return Categorical(torch.tensor(x)).entropy().item()

def qubit2xy(
    board: torch.Tensor,
    qubit: int
):
    return (board == qubit).nonzero()[0]

def position2xy(
    board: torch.Tensor,
    position: int
):
    return position // board.size(1), position % board.size(1)

class NeutralAtomsEnv2:
    """
    Neutral Atoms environment 2.
        action is (nqubits, board_size).reshape(-1)
    """
    
    def __init__(self, tasks, task_spec, actions=[], previous_correctness=0):
        self.task_spec = task_spec
        self.feature_type = task_spec.feature_type
        self.original_board = task_spec.original_board
        self.board_size = task_spec.board_size
        self.board_dim = task_spec.board_dim
        self.gate_action = task_spec.gate_action_idx
        self.num_qubits = task_spec.nqubits
        self.num_tasks = task_spec.num_tasks
        self.budget = task_spec.budget
        self.feature_count = task_spec.feature_count
        self.tasks = tasks
        self.actions = actions
        self.previous_correctness = previous_correctness
        self.reward = Reward(nvib_coeff=task_spec.nvib_coeff)
    
    @property
    def tasks_done(self):
        return self.actions.count(self.gate_action)
    
    @property
    def curr_task(self):
        return self.tasks[self.tasks_done]
    
    @property
    def non_gate_actions(self):
        return [e for e in self.actions if e != self.gate_action]
    
    @cached_property
    def board(self):
        b = self.original_board.clone()
        for action in self.non_gate_actions:
            b = self.apply_action_to_board(b, action)
        return b

    @property
    def atom_map(self):
        return {k:v for k,v in enumerate(
            torch.nonzero(self.board.view(-1)+1)[
                torch.sort(
                    self.board[torch.nonzero(self.board+1,as_tuple=True)]
                ).indices
            ].squeeze().tolist()
        )}
    
    @property
    def kh_board(self):
        # TODO: change to cached_property
        points = [Point(i,j) for i in range(self.board_dim[0]) for j in range(self.board_dim[1])]
        return Board(points, self.atom_map)
    
    @property
    def turn_actions(self):
        return list(takewhile(lambda x: x != self.gate_action, self.actions[::-1]))[::-1]
    
    @property
    def turn_moves(self):
        moves = []
        for action in self.turn_actions:
            move = self.action2move(action)
            moves.append(move)
        return moves
    
    @property
    def features(self):
        kh_board = self.kh_board
        features = []
        for player in range(self.num_qubits):
            state = [self.reward.move_gain_vector(player, kh_board, self.turn_moves),
                     self.reward.nvib_gain_vector(player, self.count_player_gates(player), kh_board)]
            for td in range(self.feature_count):
                twoq_gates = self.tasks[self.tasks_done+td] if self.tasks_done+td < self.num_tasks else []
                gain_vector = self.reward.gate_gain_vector(player, kh_board, twoq_gates, mask=True)
                state.append(gain_vector)
            features.append(torch.tensor(state))
        return features
    
    def count_player_gates(self, player):
        # assume all gates are 2q gates
        return sum(map(lambda x: player in x, chain.from_iterable(self.tasks[self.tasks_done:])))
    
    def step(self, action):
        self.actions = self.actions + [action]
        if action != self.gate_action:
            self.apply_action_to_board(self.board, action)
        return self.observation(), self.correctness_reward()
    
    def observation(self):
        if self.feature_type == 'kohei':
            return { 'features': torch.vstack(self.features) }
        elif self.feature_type == 'basic':
            flat_board = self.board.clone().flatten().long()
            flat_board[flat_board == -1] = self.num_qubits
            atom_map = self.atom_map
            grid = torch.cat((
                torch.eye(self.num_qubits),
                torch.zeros(1,self.num_qubits), # empty features
            )).index_select(0,flat_board).repeat(self.num_tasks+1,1,1)
            for i, task in enumerate(self.tasks):
                for x, y in task:
                    grid[i+1,atom_map[x],y] = 1
                    grid[i+1,atom_map[y],x] = 1
            return { 'features': grid }
        elif self.feature_type == 'fake':
            return { 'features': torch.zeros(2,2) }
        else:
            assert False, f"Unknown feature type: {self.feature_type}"
        
    def correctness_reward(self) -> float:
        """
        Computes a reward based on the correctness of the output:
        1. entropy of 
        2. parallel movement cost for reconfiguration
        
        TODO: this only considers fixed number of gates / tasks
        """
        gate_execs = self.parallel_move_executions(self.tasks, self.actions)
        curr_gate_execs = sum([len(moves) for moves in gate_execs])
        cost_lb = len(self.tasks) # cost lower bound
        if self.task_spec.reward_type == 'v2':
            cost_ub = self.budget
            rcfg_execs = self.num_rcfg_executions()
            correctness = cost_ub - curr_gate_execs - rcfg_execs
        else:
            cost_ub = len(list(chain.from_iterable(self.tasks)))
            correctness = cost_ub - curr_gate_execs
            if self.task_spec.reward_type == 'v1.5':
                max_entropy = sum(list(map(entropy, [len(x)*[1] for x in self.tasks])))
                curr_entropy = sum(list(map(entropy,[[y.count() for y in x]+(l-len(x))*[0] for x,l in zip(gate_execs,[len(x) for x in self.tasks])])))
                cost_ub += max_entropy
                correctness = cost_ub - curr_gate_execs - curr_entropy
        
        max_correctness = cost_ub - cost_lb
        all_correct = correctness == max_correctness
        correctness /= max_correctness # normalization
        reward = self.task_spec.correctness_reward_weight * (
            correctness - self.previous_correctness
        )
        self.previous_correctness = correctness

        # Bonus for fully correct programs
        reward += self.task_spec.correct_reward * all_correct

        return reward*1.0
    
    def latency_reward(self) -> float:
        """
        Computes reward for end of game based on:
        1. cost for movement during reconfiguration
        2. cost for movement during gate executions
        TODO: untested
        """
        latency_reward = self.budget \
            - self.task_spec.travel_distance_weight * (
                sum([self.action2move(action).dist() for action in self.non_gate_actions]) \
                + self.gate_exec_dist()
            )
        return latency_reward
    
    def is_correct(self) -> bool:
        current_execs = sum(self.num_gate_executions(self.tasks, self.actions))
        max_execs = len(list(chain.from_iterable(self.tasks)))
        min_execs = len(self.tasks)
        max_correctness = max_execs - min_execs
        correctness = max_execs - current_execs
        return correctness == max_correctness
        
    def legal_actions(self) -> list[int]:
        num_gate_actions = len([a for a in self.actions if a == self.gate_action])
        if len(self.actions) >= self.budget:
            assert len(self.actions) <= self.budget
            return []
        if num_gate_actions >= self.num_tasks:
            assert num_gate_actions <= self.num_tasks
            return []
        if len(self.non_gate_actions) >= (self.budget - self.num_tasks):
            assert len(self.non_gate_actions) <= (self.budget - self.num_tasks)
            return [0] # gate action 
        
        # movable mask
        mmask = (self.board+1).bool().view(-1).repeat(self.num_qubits,1)
        
        # TODO test this: threshold mask
        # tmask = torch.stack(self.features).sum(dim=1) < self.task_spec.threshold
        # assert mmask.shape == tmask.shape
        # valid_moves = (mmask + tmask).logical_not().view(-1).long().nonzero().squeeze().tolist()
        
        valid_moves = torch.cat((torch.zeros(1).long(),mmask.logical_not().view(-1))).nonzero().squeeze().tolist()
        if type(valid_moves) == int:
            valid_moves = [valid_moves]
        return [0]+valid_moves
    
    def clone(self):
        return NeutralAtomsEnv2(
            self.tasks,
            self.task_spec,
            self.actions.copy(),
            self.previous_correctness
        )
    
    # Helper
    def action2qp(self, action):
        assert action > 0
        action -= 1 # 0 was gate action
        qubit = action // self.board_size
        position = action % self.board_size
        return qubit, position
    
    def action2move(self, action):
        qubit, position = self.action2qp(action)
        src, dst = qubit2xy(self.board, qubit), position2xy(self.board, position)
        return Move(Point(*src), Point(*dst))
    
    def apply_action_to_board(
        self,
        board: torch.Tensor,
        action: int,
    ):
        qubit, position = self.action2qp(action)
        (src_x, src_y), (dst_x, dst_y) = qubit2xy(board, qubit), position2xy(board, position)
        board[src_x, src_y] = -1
        board[dst_x, dst_y] = qubit
        return board

    def get_actions_per_task(self, tasks, actions):
        actions_per_task = [None] * len(tasks)
        if actions is not None:
            actions = actions.copy()
            num_gate_actions = len([a for a in actions if a == self.gate_action])
            assert num_gate_actions <= self.num_tasks
            actions += (self.num_tasks - num_gate_actions) * [self.gate_action]
            assert len(tasks) == len([a for a in actions if a == self.gate_action])
            assert actions[-1] == self.gate_action
            actions_tensor = torch.tensor(actions)
            splits = (actions_tensor == self.gate_action).nonzero().view(-1)
            actions_per_task = torch.tensor_split(actions_tensor, splits)[:-1]
            assert len(actions_per_task) == len(tasks)
        return actions_per_task

    def parallel_move_executions(self, tasks, actions=None):
        """Pads actions with gate actions to calculate correctness reward"""
        board = self.original_board.clone() # original board
        actions_per_task = self.get_actions_per_task(tasks, actions)
        gate_execs = []
        for idx, component in enumerate(tasks):
            moves = []
            if actions is not None:
                for action in actions_per_task[idx].tolist():
                    if action == self.gate_action:
                        continue
                    board = self.apply_action_to_board(board, action)
            for gate in component:
                src, dst = gate
                src_xy, dst_xy = qubit2xy(board, src), qubit2xy(board, dst)
                moves.append(Move(Point(*src_xy), Point(*dst_xy)))
            gate_execs.append(ParallelGrouping.to_groups(moves))
        return gate_execs
    
    def num_gate_executions(self, tasks, actions=None):
        return [len(moves) for moves in self.parallel_move_executions(tasks, actions)]
    
    def num_rcfg_executions(self):
        """Pads actions with gate actions to calculate correctness reward"""
        board = self.original_board.clone() # original board
        actions_per_task = self.get_actions_per_task(self.tasks, self.actions)
        rcfg_execs = 0
        for idx in range(len(self.tasks)):
            actions = actions_per_task[idx].tolist()
            moves = [self.action2move(a) for a in actions]
            rcfg_execs += len(ParallelGrouping.to_groups(moves))
            for action in actions:
                if action == self.gate_action:
                    continue
                board = self.apply_action_to_board(board, action)
        return rcfg_execs

    def gate_exec_dist(self):
        """Pads actions with gate actions to calculate correctness reward"""
        board = self.original_board.clone() # original board
        actions_per_task = self.get_actions_per_task(self.tasks, self.actions)
        exec_dist = []
        for idx, component in enumerate(self.tasks):
            for action in actions_per_task[idx].tolist():
                if action == self.gate_action:
                    continue
                board = self.apply_action_to_board(board, action)
            for gate in component:
                src, dst = gate
                src_xy, dst_xy = qubit2xy(board, src), qubit2xy(board, dst)
                exec_dist.append(Move(Point(*src_xy), Point(*dst_xy)).dist())
        return sum(exec_dist)
