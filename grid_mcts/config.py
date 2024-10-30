import torch
from .utils import KnownBounds, TaskSpec
from ml_collections import ConfigDict
from datetime import datetime

class NeutralAtomsConfig(object):
  """NeutralAtomsConfig configuration."""

  def __init__(
      self, config: ConfigDict, atom_map: list[int], all_tasks: list[list[list[int]]], feature_type='fake'
  ):
    
    ### Self-Play
    self.num_actors = 128  # TPU actors
    self.visit_softmax_temperature_fn = lambda steps: (
        2.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
    )
    self.max_moves = torch.inf
    self.num_simulations = config.get('num_sims', 50) # 800, 10
    self.discount = 1.0

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.03
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = config.get('explore', 1.25) # 1.25

    self.known_bounds = KnownBounds(-6.0, 6.0)

    # Training
    self.epochs = 50
    self.buffer_size = 1000
    self.td_steps = 5
    self.num_selfplay = 20
    self.trainer = trainer = ConfigDict()
    trainer.seed = 12315
    trainer.save_dir = './atom_game_mcts_checkpoints/'
    trainer.log_interval = 200
    trainer.accelerator = 'cpu'
    trainer.devices = 1
    trainer.grad_norm_clip = 1.0
    trainer.training_steps = int(200) # int(1000)
    # trainer.checkpoint_interval = 500
    # trainer.target_network_interval = 100
    # trainer.window_size = int(1e6)
    trainer.batch_size = 128 # 512
    trainer.lr = 2e-4
    
    # wandb
    self.name = name = 'alpha_atoms_array'
    self.run_name = f'{name}_'+datetime.now().strftime("run_%m%d_%H_%M")
    
    # task specs
    threshold = config.get('threshold',0)
    feature_count = config.feature_count
    nvib_coeff = config.nvib_coeff
    nqubits = config.get('nqubits', 9)
    board_dim = config.get('board_dim', (2,6))
    board_size = config.get('board_size', 12)
    # nqubits = config.n_atoms
    # board_dim = (config.row,config.col)
    # board_size = config.row*config.col
    board = torch.zeros(*board_dim).view(-1).long()-1
    board[atom_map] = torch.arange(nqubits)
    board = board.reshape(*board_dim)
    # self.all_tasks = get_all_tasks(nqubits)
    self.all_tasks = all_tasks
    self.task_spec = TaskSpec(
        threshold=threshold,
        nvib_coeff=nvib_coeff,
        nqubits=nqubits,
        budget=24,
        board_dim=board_dim,
        board_size=board_size,
        gate_swap_penalty=0.2,
        original_board=board,
        num_actions=board_size*nqubits+1,
        atom_map=atom_map,
        gate_action_idx=0,
        num_tasks=3, #TODO change to larger number
        correct_reward=1.0,
        correctness_reward_weight=config.get('r_weight', 4.0), # 2.0
        latency_reward_weight=0.5,
        latency_quantile=0.0,
        feature_count=feature_count,
        travel_distance_weight=0.01,
        feature_type = feature_type, # kohei or basic or fake
        reward_type = 'v1.5', # v1: w/o reconfiguration loss, v1.5: w/ entropy loss, v2: w/ reconfiguration loss
    )

    # nnet
    self.nnet = nnet = ConfigDict()
    nnet.v_hsize = 64
    nnet.p_hsize = 32
    nnet.mlp_depth = 2
    nnet.ema_decay = 0.995
    nnet.num_bins = 51 # 301
    nnet.value_min = -25
    nnet.value_max = 25 # TODO: set based on max_correctness + latency
    nnet.mixer_cfg = mixer_cfg = ConfigDict()
    mixer_cfg.image_size = (board_size,1)
    mixer_cfg.channels = nqubits*7
    mixer_cfg.patch_size = 1
    mixer_cfg.dim = nqubits
    mixer_cfg.depth = 4