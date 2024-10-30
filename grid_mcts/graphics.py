import sys

from matplotlib import pyplot as plt
from .utils import Point, Board, Move, ParallelGrouping, Reward
from ml_collections import ConfigDict
import numpy as np
import imageio, os
import json, time

class Visualizer:
    def __init__(self, config: ConfigDict):
        self.config = config
        self.interpolation_steps = config.get("interpolation_steps", 2)
        self.step = 0
        self.t = 0
        self.min_t = config.get("min_t", 0)
        self.max_t = config.get("max_t", sys.maxsize)
        self.original = None
        self.scenarios = None
        self.point_to_id = None
        self.filenames = []

    def run(self, filename):
        self.original, self.scenarios, self.point_to_id = self.read(filename)
        b = self.original
        start_map = b.get_atom_position_map()
        for scenario in self.scenarios:
            for board in scenario["boards"]:
                end_map = board.get_atom_position_map()
                if self.t >= self.min_t:
                    self.add_configuration_frames(start_map, end_map, scenario["gates"])
                start_map = end_map
                b = board
            if self.t >= self.min_t:
                self.add_gate_frames(b, Reward.parallel_moves(b, twoq_gates=scenario["gates"]), scenario["gates"])
            self.t += 1
            self.step = 0
            if self.t == self.max_t:
                return

    def order(self, pms):
        tos = set()
        for pm in pms:
            for m in pm.moves:
                if m.frm in tos:
                    return reversed(pms)
                tos.add(m.to)
        return pms

    def read(self, filename):
        moves = []
        players = []
        map = {}
        scenarios = []
        point_to_id = {}
        state_map = {}
        move_map = {}
        with open(filename) as f:
            for l in f.readlines():
                item = json.loads(l.rstrip())
                if item["category"] == "board":
                    points = []
                    for index, p in enumerate(item["points"]):
                        point = Point(p[0], p[1])
                        points.append(point)
                        point_to_id[point] = index
                    atom_map = {}
                    for k, v in item["atom_map"].items():
                        atom_map[int(k)] = v
                    original = Board(points, atom_map)
                    board = original
                elif item["category"] == "move":
                    move = Move(Point(item["from"][0], item["from"][1]), Point(item["to"][0], item["to"][1]))
                    moves.append(move)
                    if move in map:
                        map[move].append(item["player"])
                    else:
                        map[move] = [item["player"]]
                    move_map[item["player"]] = move
                elif item["category"] == "state":
                    players.append(item["player"])
                    state_map[item["player"]] = item["state"]
                elif item["category"] == "gate":
                    # pms = ParallelGrouping.to_groups(moves)
                    # parallels = []
                    # for pm in self.order(pms):
                    #     for m in pm.moves:
                    #         board = board.transform(map[m], point_to_id[m.to])
                    #     parallels.append(board)
                    parallels = []
                    for m in moves:
                        board = board.transform(map[m].pop(0), point_to_id[m.to])
                        parallels.append(board)
                    scenario = {"category": "move", "gates": item["gates"], "boards": parallels, "states": state_map,
                                "moves": move_map, "players": players}
                    scenarios.append(scenario)
                    moves = []
                    players = []
                    map = {}
                    move_map = {}
                    state_map = {}
        return original, scenarios, point_to_id

    def save(self, path=None):
        if path is None:
            path = f'{self.config.get("tmp_dir")}/animation-{int(time.time())}.gif'
        with imageio.get_writer(f'./{path}', mode='I', fps=10) as writer:
            for filename in self.filenames:
                image = imageio.imread(filename)
                if 'step_0' in filename:
                    for _ in range(5):
                        writer.append_data(image)
                else:
                    writer.append_data(image)

            for filename in self.filenames:
                os.remove(filename)
        return f'{path}'

    def add_configuration_frames(self, start_map, end_map, next_scenario):
        self.add_frame(start_map, end_map, next_scenario, "Reconfiguration")

    def add_gate_frames(self, board, parallel_moves, next_scenario):
        position_to_atom = {}
        for atom_id, position in board.get_atom_position_map().items():
            position_to_atom[position] = atom_id
        for parallel_move in parallel_moves:
            start_map = board.get_atom_position_map()
            end_map = start_map.copy()
            move_atoms = set()
            for move in parallel_move.moves:
                atom_id = position_to_atom[move.frm]
                end_map[atom_id] = move.to
                move_atoms.add(atom_id)
            self.add_frame(start_map, end_map, next_scenario, "Gates")
            self.add_frame(end_map, start_map, next_scenario, "Gates")

    def interpolate_positions(self, start_pos, end_pos):
        y_offset = 0
        steps = self.interpolation_steps
        if start_pos != end_pos:
            y_offset = 0.4
        # Linearly interpolate between two positions over 'steps' increments
        x_positions = np.linspace(start_pos.x, start_pos.x + y_offset, steps)
        x_positions = np.hstack([x_positions, np.linspace(start_pos.x + y_offset, end_pos.x + y_offset, steps)])
        x_positions = np.hstack([x_positions, np.linspace(end_pos.x + y_offset, end_pos.x, steps)])
        x_positions = np.hstack([x_positions, np.linspace(end_pos.x, end_pos.x, steps)])

        y_positions = np.linspace(start_pos.y, start_pos.y + y_offset, steps)
        y_positions = np.hstack([y_positions, np.linspace(start_pos.y + y_offset, end_pos.y + y_offset, steps)])
        y_positions = np.hstack([y_positions, np.linspace(end_pos.y + y_offset, end_pos.y, steps)])
        y_positions = np.hstack([y_positions, np.linspace(end_pos.y, end_pos.y, steps)])

        return x_positions, y_positions

    def draw_gain_board(self):
        b = self.original
        fig, ax = plt.subplots(figsize=(self.original.width(), self.original.height()))
        for scenario in self.scenarios:
            if self.t >= self.min_t and self.t < self.max_t:
                for player in scenario["players"]:
                    vector = None
                    states = scenario["states"]
                    if player not in states:
                        continue
                    for state in states[player]:
                        if vector is None:
                            vector = np.array(state, dtype=np.float32)
                        else:
                            vector += np.array(state)
                    self.add_gain_board(ax, player, b, vector)
                    filename = f'{self.config.get("tmp_dir")}/gain_{player}_{self.t}.png'
                    fig.savefig(filename, dpi=100)
                    plt.close(fig)
                    self.filenames.append(filename)
                    if player in scenario["moves"]:
                        move = scenario["moves"][player]
                        b = b.transform(player, self.point_to_id[move.to])

            for board in scenario["boards"]:
                b = board
            self.t += 1
            self.step = 0

    def add_gain_board(self, ax, player, board, gain_vector):
        ax.clear()
        fig_size = self.original.fig_size()
        ax.set_xlim(fig_size[0], fig_size[1])
        ax.set_ylim(fig_size[2], fig_size[3])

        self.draw_base(ax)
        for id, position in board.get_atom_position_map().items():
            ax.annotate(str(id).zfill(2), (position.x, position.y),
                        bbox={"boxstyle": "circle,pad=0.5", "facecolor": "white"},
                        fontsize=8)
        for j, point in enumerate(board.points):
            value = "{:.1f}".format(gain_vector[j])
            ax.annotate(f'{value}', (point.x + 0.2, point.y + 0.2), color="green", fontsize=10)
        ax.text(0, 1.05, f'Step: {self.t} Player: {player}', transform=ax.transAxes, fontsize=12)
        return ax.figure

    def draw_base(self, ax):
        for point in self.original.points:
            ax.annotate('.', (point.x, point.y), bbox={"boxstyle": "circle,pad=0.1"}, fontsize=2)

    def draw_interpolated_frame(self, ax, move_atoms, positions, next_scenario, others):
        ax.clear()
        fig_size = self.original.fig_size()
        ax.set_xlim(fig_size[0], fig_size[1])
        ax.set_ylim(fig_size[2], fig_size[3])

        self.draw_base(ax)
        for id, (x, y) in zip(self.original.atom_map.keys(), positions):
            ax.annotate(str(id).zfill(2), (x, y),
                        bbox={"boxstyle": "circle,pad=0.5", "facecolor": "white"},
                        fontsize=8)
            if id in move_atoms:
                ax.plot([fig_size[0], fig_size[1]], [y, y], linewidth=0.5, color='blue')
                ax.plot([x, x], [fig_size[2], fig_size[3]], linewidth=0.5, color='blue')
        ax.text(0, 1.05, f'Step: {self.t} ({others})', transform=ax.transAxes, fontsize=12)
        ax.text(0, -0.25, f'next gates: {str(next_scenario)}', transform=ax.transAxes, fontsize=8)
        return ax.figure

    def add_frame(self, start_map, end_map, next_scenario, label):
        reconfiguration_positions_map = {}
        move_atoms = set()
        for k, pos in start_map.items():
            if end_map[k] != pos:
                move_atoms.add(k)

        for q_id in self.original.atom_map.keys():
            step = self.step
            start_pos = start_map[q_id]
            end_pos = end_map[q_id]
            xs, ys = self.interpolate_positions(start_pos, end_pos)[0], \
                self.interpolate_positions(start_pos, end_pos)[1]
            for x, y in zip(xs, ys):
                if step not in reconfiguration_positions_map:
                    reconfiguration_positions_map[step] = []
                reconfiguration_positions_map[step].append((x, y))
                step += 1

        for step, reconfiguration_positions in reconfiguration_positions_map.items():
            fig, ax = plt.subplots(figsize=(self.original.width()+3, self.original.height()+3))
            self.draw_interpolated_frame(ax, move_atoms, reconfiguration_positions, next_scenario, label)
            plt.subplots_adjust(bottom=0.2)
            filename = f'{self.config.get("tmp_dir")}/easy_step_{step}_{self.t}.png'
            fig.savefig(filename, dpi=100)
            plt.close(fig)
            self.filenames.append(filename)
            self.step += 1
