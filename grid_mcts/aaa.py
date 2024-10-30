"""Alpha Atoms Array"""
import torch
from lightning import Fabric
from .game import Game
from .mcts import play_game
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage


class AlphaAtomsArrayTask:

    def __init__(self, network, config) -> None:
        super().__init__()
        self.config = config
        self.task_spec = config.task_spec
        self.network = network
        self.replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(config.buffer_size))
        self.selfplay_iter = 0

    def run_selfplay(self):
        avg = torch.zeros(self.config.num_selfplay)
        print('tasks:')
        print(self.config.all_tasks)
        print('\n'.join([' '.join(f'{int(y):>3d}' if y >= 0 else '   ' for y in x) for x in (self.task_spec.original_board).tolist()]))
        for idx in range (self.config.num_selfplay):
            game = Game(
                self.config.all_tasks,
                self.task_spec.num_actions,
                self.config.discount,
                self.task_spec
            )
            game = play_game(game, self.config, self.network)
            self.save_game(game)
            avg[idx]=game.game_total_execs()
        print(f"selfplay iter: {self.selfplay_iter+1}, avg num execs: {avg.mean()}")
        return game.history # last game historyfor visualization
    
    def save_game(self, game):
        num_samples = len(game.history)
        td = self.config.td_steps
        features = []
        bootstrap_features = []
        cvals = []
        lvals = []
        pis = []
        bvals = []
        for i in range(len(game.history)):
            obs = game.make_observation(i)
            bootstrap_obs = game.make_observation(i+td)
            features.append(obs['features'])
            bootstrap_features.append(bootstrap_obs['features'])
            cval, lval, pi, bval = game.make_target(i,td,-1)
            cvals.append(cval)
            lvals.append(lval)
            pis.append(pi)
            bvals.append(bval)
        features = torch.stack(features).float()
        bootstrap_features = torch.stack(bootstrap_features).float()
        cvals = torch.tensor(cvals).float()
        lvals = torch.tensor(lvals).float()
        pis = torch.tensor(pis).float()
        bvals = torch.tensor(bvals).float()
        observations = TensorDict({
            ('obs', 'features'): features,
            ('bootstrap_obs', 'features'): bootstrap_features,
            ('target', 'correctness_values'): cvals,
            ('target', 'latency_values'): lvals,
            ('target', 'policies'): pis,
            ('target', 'bootstrap_discounts'): bvals,
        },
        batch_size=num_samples)
        self.replay_buffer.extend(observations)
    
    def fit(self, cfg, logger):
        fabric = Fabric(accelerator=cfg.accelerator, devices=cfg.devices, loggers=[logger])
        fabric.seed_everything(cfg.seed)
        fabric.launch()

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=cfg.lr)
        model, optimizer = fabric.setup(self.network, optimizer)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.selfplay_iter == 0:
            print(f"total trainable params: {pytorch_total_params}")
        model.train()
        optimizer.zero_grad()
        for iteration in range(cfg.training_steps):
            batch = self.replay_buffer.sample(cfg.batch_size)
            loss = model(batch)
            fabric.backward(loss)
            fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
            if (iteration+1) % cfg.log_interval == 0:
                fabric.log_dict({"train/loss": loss})
                print(f"iteration: {iteration+1}, loss: {loss:.2f}")
            optimizer.step()
            model.t_nnet.update(model.parameters()) # update target network
            optimizer.zero_grad()
            
        state = { "model": model, "optimizer": optimizer }
        fabric.save(cfg.save_dir+f"aaa_checkpoint_{self.selfplay_iter}.ckpt", state)
        self.selfplay_iter += 1

