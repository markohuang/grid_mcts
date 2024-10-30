import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from ml_collections import ConfigDict
from torch_ema import ExponentialMovingAverage
from .nnet import NeutralAtomsMLP, NeutralAtomsMLP2, FakeNet
from .utils import TaskSpec, NetworkOutput
from typing import Dict


class Network(pl.LightningModule):
	"""Wrapper around Representation and Prediction networks."""

	def __init__(self, nnet_config: ConfigDict, task_spec: TaskSpec):
		super().__init__()
		self.feature_type = task_spec.feature_type
		self.cfg = cfg = nnet_config
		self.action_space_size = task_spec.num_actions
		if self.feature_type == 'fake':
			self.nnet = FakeNet(cfg, task_spec)
			self.t_nnet = FakeNet(cfg, task_spec)
		elif self.feature_type == 'kohei':
			self.nnet = NeutralAtomsMLP(cfg, task_spec)
			self.t_nnet = ExponentialMovingAverage(self.nnet.parameters(), decay=cfg.ema_decay)
		elif self.feature_type == 'basic':
			self.nnet = NeutralAtomsMLP2(cfg, task_spec)
			self.t_nnet = ExponentialMovingAverage(self.nnet.parameters(), decay=cfg.ema_decay)
		self.register_buffer('categories', torch.linspace(self.cfg.value_min,self.cfg.value_max,steps=self.cfg.num_bins)[:,None])
      
	# @torch.inference_mode()
	def inference(self, observation: Dict, aslist=False) -> NetworkOutput:
		features = observation['features']
		assert features.dim() >= 2
		if self.feature_type != 'basic' and features.dim() == 2:
			features = features[None,:] # add batch dimension
		elif self.feature_type == 'basic' and features.dim() == 3:
			features = features[None,:]
		output = self.nnet(features)
		correctness_logits, latency_logits, pi = output
		correctness_mean = self.logits2values(correctness_logits, self.categories)
		latency_mean = self.logits2values(latency_logits, self.categories)
		pi = pi.squeeze() if not aslist else pi.squeeze().tolist()
		return NetworkOutput(
			value=correctness_mean + latency_mean,
			correctness_value_logits=correctness_logits.squeeze(),
			latency_value_logits=latency_logits.squeeze(),
			policy_logits=pi,
		)
      
	def forward(self, batch):
		predictions = self.inference(batch['obs'])
		with self.t_nnet.average_parameters(), torch.no_grad(): # use target network
			bootstrap_predictions = self.inference(batch['bootstrap_obs'])
		target_correctness, target_latency, target_policy, bootstrap_discount = (
				batch['target'].values()
		)
		target_correctness += (
			bootstrap_discount * self.logits2values(
				bootstrap_predictions.correctness_value_logits,
				self.categories
			)
		).clip(self.cfg.value_min, self.cfg.value_max)
		target_correctness = (
			(1 - bootstrap_discount) * target_correctness +\
			0.5 * bootstrap_discount *\
			(
       			target_correctness +\
				self.logits2values(
        			bootstrap_predictions.correctness_value_logits,
           			self.categories
            	)
			)
		)
		loss = F.cross_entropy(predictions.policy_logits, target_policy)
		loss += F.cross_entropy(
			predictions.correctness_value_logits,
     		self.to_onehot(target_correctness)
		)
		loss += F.cross_entropy(
     		predictions.latency_value_logits,
        self.to_onehot(target_latency)
    	)
		return loss.mean()

	def logits2values(self, logits, categories):
		return (torch.exp(logits) @ categories).squeeze()
 
	def to_onehot(self, val):
		buckets = self.categories.squeeze()
		return F.one_hot(
				torch.bucketize(val, buckets),
				num_classes=self.cfg.num_bins
		).float()

	def training_steps(self) -> int:
		return self.global_step
