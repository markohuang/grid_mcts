import torch, json

def log4visualizer(all_tasks, task_specs, history, tmp_path='test_ml.txt'):
    board_size = task_specs.board_size
    board_dim = task_specs.board_dim
    atom_map = task_specs.atom_map
    gate_action = task_specs.gate_action_idx
    board = task_specs.original_board.clone()
    logs = []
    def action2qp(action, board_size=board_size):
        assert action > 0
        action -= 1 # 0 was gate action
        qubit = action // board_size
        position = action % board_size
        return qubit, position
        
    def apply_action_to_board(
        board: torch.Tensor,
        action: int,
    ):
        qubit, position = action2qp(action)
        (src_x, src_y), (dst_x, dst_y) = qubit2xy(board, qubit), position2xy(board, position)
        board[src_x, src_y] = -1
        board[dst_x, dst_y] = qubit
        return board

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


    logs.append(json.dumps(dict(
        category    = "board",
        points      = [[i,j] for i in range(board_dim[0]) for j in range(board_dim[1])],
        atom_map    = {str(k): v for k, v in enumerate(atom_map)},
    )))

    curr_gate_idx = 0
    for action in history:
        if action == gate_action:
            logs.append(json.dumps({
                "category": "gate",
                "gates": all_tasks[curr_gate_idx],
                "reward": -12315
            }))
            curr_gate_idx += 1
            continue
        qubit, position = action2qp(action)
        (src_x, src_y), (dst_x, dst_y) = qubit2xy(board, qubit), position2xy(board, position)
        logs.append(json.dumps({
            "category": "move",
            "player": qubit,
            "from": (src_x.item(), src_y.item()),
            "to": (dst_x, dst_y),
            "reward": -12315,
        }))
        apply_action_to_board(board, action)


    with open(tmp_path, 'w') as f:
        for line in logs:
            f.write(f'{line}\n')
            
            