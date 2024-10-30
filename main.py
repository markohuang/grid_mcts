import json
import torch, random
from grid_mcts.game import Game
from grid_mcts.mcts import play_game
from grid_mcts.config import NeutralAtomsConfig
from grid_mcts.network import Network
from grid_mcts.graphics import Visualizer
from ml_collections import ConfigDict
from visualization import log4visualizer
            
if __name__ == "__main__":
    import argparse, os
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--r_weight", type=float, default=4.0) # best was 4.0
    parser.add_argument("--reward_type", type=str, default='v1.5') # v1.5
    parser.add_argument("--num_sims", type=int, default=200) # best was 10000
    parser.add_argument("--explore", type=float, default=1.5) # best was 2.0
    parser.add_argument("--map_num", type=int, default=0)
    args = parser.parse_args()
    seed = 12315
    # random.seed(seed)
    # torch.manual_seed(seed)
    
    maps = [
        ( # 2x6 with 9 atoms
            ((2,6), 9),
            [11, 3, 1, 8, 10, 7, 5, 6, 4], 
            [[[0,8],[6,2],[1,7],[3,5]],[[0,2],[6,8],[1,5],[3,7]],[[3,7],[1,5],[6,8]]]
        ),
        ( # 4x4 with 8 atoms
            ((4,4), 8),
            [5, 9, 10, 11, 2, 6, 0, 14],
            [[[0, 1], [2, 3], [5, 4], [6, 7]], [[0, 1], [2, 3], [5, 4], [6, 7]], [[0, 1], [2, 3], [5, 4], [6, 7]]]
        ),
        ( # 5x5 with 12 atoms
            ((5,5), 12),
            [4, 12, 7, 11, 15, 5, 2, 21, 22, 23, 20, 8],
            [[[4, 3], [6, 7], [9, 8], [10, 11]], [[2, 1], [4, 5], [7, 6], [8, 9]], [[0, 1], [2, 3], [6, 5], [9, 8]]]
        )
    ]
    
    mcfg, atom_map, all_tasks = maps[args.map_num]
    
    cfg = ConfigDict()
    cfg["n_atoms"] = mcfg[1]
    cfg["nqubits"] = mcfg[1]
    cfg["feature_count"] = 5
    cfg["row"] = mcfg[0][0]
    cfg["col"] = mcfg[0][1]
    cfg["nvib_coeff"] = -0.001
    cfg["seed"] = 0
    cfg["board_dim"] = mcfg[0]
    sz = cfg['col']*cfg['row']
    cfg["board_size"] = sz
    
    cfg.update(**{k:v for k,v in vars(args).items() if v is not None})
    cfg.name = 'mcts_only'
    cfg.root_dir = './'
    run_param_str = f'r{cfg.r_weight}_sims{cfg.num_sims}_exp{cfg.explore}'
    cfg.run_name = run_param_str+datetime.now().strftime("run_%m%d_%H_%M")
    cfg.save_dir = f"{cfg.root_dir}/{cfg.name}_checkpoints/{cfg.run_name}/"
    print('run_name', cfg.run_name)
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # atom_map = torch.randperm(sz)[:cfg['n_atoms']].tolist()
    config = NeutralAtomsConfig(cfg, atom_map, all_tasks, 'basic')
    config.num_selfplay = 1 # TODO: change to 20
    network = Network(config.nnet, config.task_spec)
    
    
    
    import pathlib
    # from grid_mcts.config import NeutralAtomsConfig
    # from grid_mcts.network import Network
    from grid_mcts.aaa import AlphaAtomsArrayTask
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
    torch.set_float32_matmul_precision('medium')
    cfg = config
    # # atom_map = torch.randperm(12)[:9].tolist()
    # atom_map = atom_map = [11, 3, 1, 8, 10, 7, 5, 6, 4]
    # all_tasks = [[[0, 8], [6, 2], [1, 7], [3, 5]], [[0, 2], [6, 8], [1, 5], [3, 7]], [[3, 7], [1, 5], [6, 8]]]
    task = AlphaAtomsArrayTask(network, cfg)
    file_path = str(pathlib.Path(__file__).parent) + '/atom_game_mcts_logs'
    logger = CSVLogger(
        save_dir=file_path,
        name=cfg.run_name,
    )
    os.makedirs('./ml_gif_results/', exist_ok=True)
    for _ in range(cfg.epochs):
        last_game_history = task.run_selfplay()
        num_gate_actions = len([a for a in last_game_history if a == config.task_spec.gate_action_idx])
        assert num_gate_actions <= len(all_tasks)
        last_game_history += (len(all_tasks) - num_gate_actions) * [config.task_spec.gate_action_idx]
        log4visualizer(all_tasks, config.task_spec, last_game_history)
        run_name = run_param_str+datetime.now().strftime("run_%m%d_%H_%M")
        tmp_dir = f'./ml_gif_results/{cfg.run_name}'
        os.makedirs(tmp_dir, exist_ok=True)
        vis = Visualizer(dict(tmp_dir=tmp_dir,interpolation_steps=5))
        vis.run('test_ml.txt')
        vis.save()
        task.fit(cfg.trainer, logger)

