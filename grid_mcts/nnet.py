import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from functools import partial


class FakeNet:
    def __init__(self, cfg, task_specs):
        self.num_bins = cfg.num_bins
        self.num_actions = task_specs.num_actions
        
    def __call__(self, *args, **kwargs):
        val = torch.ones(1,self.num_bins)
        pi = torch.ones(1,self.num_actions)
        return F.log_softmax(val, dim=1), F.log_softmax(val, dim=1), F.log_softmax(pi, dim=1)

    def update(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    

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


class NeutralAtomsMLP(nn.Module):
    def __init__(self, cfg, task_specs):
        super(NeutralAtomsMLP, self).__init__()
        self.cfg = cfg
        self.task_specs = task_specs
        self.mixer1 = MLPMixer(**cfg.mixer_cfg) # value  network
        self.mixer2 = MLPMixer(**cfg.mixer_cfg) # policy network
        self.W_correctness  = nn.Linear(task_specs.board_size, cfg.num_bins, bias=False)
        self.W_latency      = nn.Linear(task_specs.board_size, cfg.num_bins, bias=False)
        self.W_gate_action  = nn.Linear(task_specs.board_size, 1, bias=False)
        self.W_value        = nn.Linear(task_specs.nqubits, 1, bias=False)
        self.W_pi           = nn.Linear(task_specs.nqubits, 1, bias=False)
        self.W_correctness.weight.data /= 100
        self.W_latency.weight.data /= 100
        self.W_gate_action.weight.data /= 100
        self.W_value.weight.data /= 100
        self.W_pi.weight.data /= 100
        
    def forward(self, features):
        val = self.mixer1(features.unsqueeze(-1)) # -> b, h*w, players
        pi = self.mixer2(features.unsqueeze(-1)) # -> b, h*w, players
        val_board = self.W_value(val).squeeze(-1) # b, h*w
        pi_board = self.W_pi(pi).squeeze(-1) # b, h*w
        cval = self.W_correctness(val_board)
        lval = self.W_latency(val_board)
        gate_action = self.W_gate_action(pi_board)
        pi = torch.cat((gate_action, pi.permute(0,2,1).flatten(start_dim=1)),dim=-1)
        return F.log_softmax(cval, dim=1), F.log_softmax(lval, dim=1), F.log_softmax(pi, dim=1)


class ValueNetwork(nn.Module):
    def __init__(self, cfg, task_specs):
        super(ValueNetwork, self).__init__()
        self.cfg = cfg
        self.nqubits = task_specs.nqubits
        self.board_size = task_specs.board_size
        self.ntasks = task_specs.num_tasks+1 # blank task for reference
        self.mixer1 = MLPMixer(
            channels    = self.board_size,
            depth       = cfg.mlp_depth,
            dim         = cfg.v_hsize,
            image_size  = (self.nqubits,1),
            patch_size  = 1
        ) # every task goes through same MLP
        self.mixer2 = MLPMixer(
            channels    = self.nqubits*cfg.v_hsize,
            depth       = cfg.mlp_depth,
            dim         = cfg.v_hsize,
            image_size  = (self.ntasks,1),
            patch_size  = 1
        )
        self.W_correctness  = nn.Linear(self.ntasks*cfg.v_hsize, cfg.num_bins)
        self.W_latency      = nn.Linear(self.ntasks*cfg.v_hsize, cfg.num_bins)
        self.W_gate_action  = nn.Linear(self.ntasks*cfg.v_hsize, 1)
        self.W_correctness.weight.data  /= 100
        self.W_correctness.bias.data    /= 100
        self.W_latency.weight.data      /= 100
        self.W_latency.bias.data        /= 100
        self.W_gate_action.weight.data  /= 100
        self.W_gate_action.bias.data    /= 100
        self.softplus = nn.Softplus()
        
    def forward(self, grids):
        bs, ntasks = grids.shape[:2]
        x = self.mixer1(grids.flatten(0,1).unsqueeze(-1)) # b*ntasks x nqubits x dim
        x = self.mixer2(x.reshape(bs, ntasks, -1).permute(0,2,1).unsqueeze(-1))
        return self.W_correctness(x.flatten(1)), self.W_latency(x.flatten(1)), self.softplus(self.W_gate_action(x.flatten(1)))

class PolicyNetwork(nn.Module):
    def __init__(self, cfg, task_specs):
        super(PolicyNetwork, self).__init__()
        self.cfg = cfg
        self.nqubits = task_specs.nqubits
        self.board_size = task_specs.board_size
        self.ntasks = task_specs.num_tasks + 1
        self.mixer1 = MLPMixer(
            channels    = self.board_size,
            depth       = cfg.mlp_depth,
            dim         = cfg.p_hsize,
            image_size  = (self.nqubits,1),
            patch_size  = 1
        )
        self.mixer2 = MLPMixer(
            channels    = self.ntasks*cfg.p_hsize,
            depth       = cfg.mlp_depth,
            dim         = cfg.p_hsize,
            image_size  = (self.nqubits,1),
            patch_size  = 1
        )
        self.W_pi = nn.Linear(cfg.p_hsize, self.board_size)
        self.W_pi.weight.data  /= 100
        self.W_pi.bias.data    /= 100
        # advice as per https://arxiv.org/pdf/2006.05990
        self.softplus = nn.Softplus()
        
    def forward(self, grids):
        bs, ntasks = grids.shape[:2]
        x = self.mixer1(grids.flatten(0,1).unsqueeze(-1)) # b*ntasks x board_size x dim
        _, nqubits, dim = x.shape
        x = self.mixer2(
                x\
                    .reshape(bs, ntasks, nqubits, dim, 1)\
                    .permute(0,1,3,2,4)\
                    .flatten(1,2)
            )
        return self.softplus(self.W_pi(x).squeeze(-1))
        
class NeutralAtomsMLP2(nn.Module):
    def __init__(self, cfg, task_specs):
        super(NeutralAtomsMLP2, self).__init__()
        self.cfg = cfg
        self.task_specs = task_specs
        self.value_net = ValueNetwork(cfg, task_specs) # value network
        self.pi_net = PolicyNetwork(cfg, task_specs) # policy network
        
    def forward(self, features):
        cval, lval, gate_action = self.value_net(features)
        pi = self.pi_net(features)
        pi = torch.cat((gate_action, pi.flatten(start_dim=1)),dim=-1)
        return F.log_softmax(cval, dim=1), F.log_softmax(lval, dim=1), F.log_softmax(pi, dim=1)
