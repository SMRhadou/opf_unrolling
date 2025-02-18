import argparse
import subprocess
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict
import copy

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.data import Data, HeteroData


import src.opf.powerflow as pf
from src.opf.constraints import equality, inequality
from src.opf.dataset import PowerflowBatch, PowerflowData
from src.opf.hetero import HeteroGCN, HeteroSage

class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # self.act = Swish()
        self.act = nn.LeakyReLU()

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(self.n_channels // 4, self.n_channels),
                                       self.act,
                                       nn.Linear(self.n_channels, self.n_channels)
                                       )
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([10000.])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb)
        return emb
    

class OPFUnrolled(pl.LightningModule):
    def __init__(self,
        model: torch.nn.Module,
        n_nodes: tuple[int, int, int],
        lr_critic=1e-4,
        lr_actor=1e-4,
        weight_decay=0.0,
        lr_dual=1e-3,
        lr_common_critic=1e-4,
        lr_common_actor=1e-4,
        weight_decay_dual=0.0,
        eps=1e-3,
        constraint_eps = 0.05,
        exploration_rate=0.4,
        enforce_Vconstraints=False,
        enforce_Sg_constraints=False,
        enforce_bus_reference=False,
        detailed_metrics=False,
        multiplier_table_length=5000,
        cost_weight=1.0,
        aux_model: torch.nn.Module = None,  # only if mode == actor
        augmented_weight: float = 0.0,
        supervised_weight: float = 0.0,
        powerflow_weight: float = 0.0,
        noise_std: float = 0.1,
        warmup: int = 0,
        supervised_warmup: int = 0,
        common: bool = True,
        forget: bool =True,
        update_common_multipliers: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "aux_model", "kwargs"])
        self.aux_model = aux_model
        self.model = model
        self.mode = kwargs['mode']
        self.lr = lr_critic if self.mode=="critic" else lr_actor
        self.weight_decay = weight_decay
        self.lr_dual = lr_dual
        self.lr_common = lr_common_critic if self.mode=="critic" else lr_common_actor
        self.weight_decay_dual = weight_decay_dual
        self.eps = eps
        self.constraint_eps = constraint_eps if self.mode == 'actor' else 0.0
        self.exploration_rate = exploration_rate
        self._enforce_Vconstraints = enforce_Vconstraints
        self._enforce_Sg_constraints = enforce_Sg_constraints
        self._enforce_bus_reference = enforce_bus_reference
        self._enforce_constraints = enforce_Vconstraints or enforce_Sg_constraints or enforce_bus_reference
        self.detailed_metrics = detailed_metrics
        self.automatic_optimization = False
        self.multiplier_table_length = multiplier_table_length
        self.cost_weight = cost_weight
        self.augmented_weight = augmented_weight
        self.supervised_weight = supervised_weight
        self.powerflow_weight = powerflow_weight
        self.noise_std = noise_std
        self.warmup = warmup
        self.supervised_warmup = supervised_warmup
        self.common = common
        self.forget = forget
        self.update_common_multipliers = update_common_multipliers
        n_bus, n_branch, n_gen = n_nodes
        self.init_multipliers(n_bus, n_gen, n_branch)
        self.init_training_multipliers()
        self.best_val = float("inf")
        if self.aux_model is not None:
            self.aux_model.eval()
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("OPFUnrolling")
        group.add_argument("--constraint_eps", type=float, default=0.1)
        group.add_argument("--exploration_rate", type=float, default=0.4)
        group.add_argument("--lr_critic", type=float, default=2.96e-4)
        group.add_argument("--lr_actor", type=float, default=1e-5)
        group.add_argument("--weight_decay", type=float, default=0.0)
        group.add_argument("--lr_dual", type=float, default=0.1)
        group.add_argument("--lr_common_critic", type=float, default=1.12e-4)
        group.add_argument("--lr_common_actor", type=float, default=1e-4)
        group.add_argument("--weight_decay_dual", type=float, default=0.0)
        group.add_argument("--eps", type=float, default=1e-3)
        # group.add_argument("--enforce_constraints", action="store_true", default=True)
        group.add_argument("--detailed_metrics", action="store_true", default=False)
        group.add_argument("--cost_weight", type=float, default=1.0)
        group.add_argument("--augmented_weight", type=float, default=0.9)
        group.add_argument("--supervised_weight", type=float, default=0.0)
        group.add_argument("--powerflow_weight", type=float, default=0.0)
        group.add_argument("--noise_std", type=float, default=0.5)
        group.add_argument("--warmup", type=int, default=0)
        group.add_argument("--supervised_warmup", type=int, default=0)
        group.add_argument("--no_common", dest="common", action="store_false", default=False)

    def init_training_multipliers(self):
        self.training_multipliers_common = Parameter(
                torch.zeros(self.model.n_layers, device=self.device)
            )
        
    def init_multipliers(self, n_bus, n_gen, n_branch):
        with torch.no_grad():
            self.multiplier_metadata = OrderedDict(
                [
                    ("equality/bus_active_power", torch.Size([n_bus])),
                    ("equality/bus_reactive_power", torch.Size([n_bus])),
                    ("equality/bus_reference", torch.Size([n_bus])),
                    ("inequality/voltage_magnitude", torch.Size([n_bus, 2])),
                    ("inequality/active_power", torch.Size([n_gen, 2])),
                    ("inequality/reactive_power", torch.Size([n_gen, 2])),
                    ("inequality/forward_rate", torch.Size([n_branch, 2])),
                    ("inequality/backward_rate", torch.Size([n_branch, 2])),
                    ("inequality/voltage_angle_difference", torch.Size([n_branch, 2])),
                ]
            )
            self.multiplier_numel = torch.tensor(
                [x.numel() for _, x in self.multiplier_metadata.items()]
            )
            self.multiplier_offsets = torch.cumsum(self.multiplier_numel, 0)
            self.n_multipliers = int(self.multiplier_offsets[-1].item())

            self.multiplier_table = {
                name: [] for name in self.multiplier_metadata.keys()
            }

            self.longterm_multiplier_table = {
                name: [] for name in self.multiplier_metadata.keys()
            }

            # multiplier_inequality_mask_list = []
            # for name, numel in zip(
            #     self.multiplier_metadata.keys(), self.multiplier_numel
            # ):
            #     if "inequality" in name:
            #         multiplier_inequality_mask_list.append(
            #             torch.ones(int(numel.item()), dtype=torch.bool)
            #         )
            #     else:
            #         multiplier_inequality_mask_list.append(
            #             torch.zeros(int(numel.item()), dtype=torch.bool)
            #         )
            # self.register_buffer(
            #     "multiplier_inequality_mask",
            #     torch.cat(multiplier_inequality_mask_list),
            #     persistent=False,
            # )
            # self.multipliers_table = torch.nn.Embedding(
            #     num_embeddings=multiplier_table_length,
            #     embedding_dim=self.n_multipliers,
            # )
            # self.multipliers_table.weight.data.zero_()
            # self.multipliers_common = Parameter(
            #     torch.zeros(self.n_multipliers, device=self.device)
            # )     

    def get_rand_multipliers(self, idx: torch.Tensor, Lmax=1) -> dict[str, torch.Tensor]:
        n_equality = self.multiplier_offsets[2]
        multipliers = torch.zeros(idx.shape[0], self.n_multipliers, device=self.device)
        # Lmax = Lmax if self.mode == 'critic' else 1 
        multipliers[:, :n_equality] = 2*Lmax * (torch.rand(idx.shape[0], n_equality, device=self.device)-0.5)
        multipliers[:, n_equality:] = Lmax * torch.rand(idx.shape[0], self.n_multipliers - n_equality, device=self.device)

        if hasattr(self, 'exploitation_dataset') and list(self.exploitation_dataset.values())[0] != []:
            # exploitation_indices = torch.randperm(len(list(self.exploitation_dataset.values())[0]))[:idx.shape[0]]
            exploration_ex = torch.rand(idx.shape[0], device=self.device) < self.exploration_rate
        
        multiplier_dict = {}
        for data, (name, shape) in zip(
            torch.tensor_split(multipliers, self.multiplier_offsets, dim=1),
            self.multiplier_metadata.items(),
        ):
            if hasattr(self, 'exploitation_dataset') and self.exploitation_dataset[name] != []:
                # assert self.mode == 'critic'
                # exploitation_indices = torch.randperm(self.exploitation_dataset[name].numel())
                # view_shape = (idx.shape[0],) + shape
                # exploration_ex_view = exploration_ex.view(-1, *([1] * len(shape)))
                # multiplier_dict[name] = (
                #     data.view(view_shape) * exploration_ex_view
                #     + (self.exploitation_dataset[name].view(-1)[exploitation_indices].view(self.exploitation_dataset[name].shape))[:idx.shape[0]].to(self.device) * ~exploration_ex_view
                # )
                assert self.mode == 'critic'
                exploitation_indices = torch.randperm(self.exploitation_dataset[name].shape[0])[:idx.shape[0]]
                exploration_ex_view = exploration_ex.view(-1, *([1] * len(shape)))
                exploitation_data = self.exploitation_dataset[name][exploitation_indices].to(self.device)
                exploitation_data = exploitation_data.view(-1)[torch.randperm(exploitation_data.numel())].view(exploitation_data.shape)
                multiplier_dict[name] = torch.where(exploration_ex_view, data.view((idx.shape[0],) + shape), exploitation_data)
            else:
                multiplier_dict[name] = data.view((idx.shape[0],) + shape)            
        return multiplier_dict           

    def project_training_multipliers_common(self):
        """
        Project the inequality multipliers to be non-negative. Only the common multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.training_multipliers_common.data = (
            self.training_multipliers_common.data.relu_()
        )

    def project_multipliers_common(self):
        """
        Project the inequality multipliers to be non-negative. Only the common multipliers are projected.
        """
        # use .data to avoid autograd tracking since we are modifying the data in place
        # no need for autograd since we are not backpropagating through this operation
        self.multipliers_common.data[self.multiplier_inequality_mask] = (
            self.multipliers_common.data[self.multiplier_inequality_mask].relu_()
        )

    def _update_multiplier_table(self, multipliers: Dict[str, Dict]):
        with torch.no_grad():
            for layer, multiplier_dict in multipliers.items():
                for name, value in multiplier_dict.items():
                    if name in self.multiplier_metadata:
                        self.multiplier_table[name] += value.detach().cpu() + self.noise_std * torch.rand_like(value.detach().cpu())
                        indices = torch.randperm(len(value))[:2]
                        self.longterm_multiplier_table[name] += value[indices].detach().cpu()
                    # if self.forget and len(self.multiplier_table[name]) > self.multiplier_table_length:
                    #     self.multiplier_table[name] = self.multiplier_table[name][-self.multiplier_table_length :]
                        self.log(f"{layer}/multipliers_dataset/{name}/max", value.detach().cpu().max())
                        self.log(f"{layer}/multipliers_dataset/{name}/mean", value.detach().cpu().mean())
                        self.log(f"{layer}/multipliers_dataset/{name}/var", value.detach().cpu().var().sqrt())
                
    
    def _generate_exploitation_dataset(self, multipliers: list[torch.Tensor], n_samples: int, longterm_multipliers: list[torch.Tensor]=None):
        """
        Generate the exploitation dataset from the multipliers.
        """
        self.exploitation_dataset = {}
        indices = torch.randperm(len(list(multipliers.values())[0]))[:n_samples]
        for name, value in multipliers.items():
            if value == []:
                self.exploitation_dataset[name] = []
            else:
                # if n_samples > len(value):
                #     n_samples = len(value)
                # randomly sample n_samples from the multipliers
                self.exploitation_dataset[name] = torch.stack([value[i] for i in indices])
                if longterm_multipliers is not None:
                    longterm_exploitation = torch.stack(longterm_multipliers[name][-20000:])
                    self.exploitation_dataset[name] = torch.cat([self.exploitation_dataset[name], 
                                                                longterm_exploitation  + self.noise_std * torch.rand_like(longterm_exploitation)])


    def forward(
        self,
        input: PowerflowBatch | PowerflowData,
        multipliers: Dict[str, torch.Tensor] | None = None,
    ) -> tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[pf.PowerflowVariables, torch.Tensor, torch.Tensor]:
              - The powerflow variables,
              - the predicted forward power,
              - and the predicted backward power.
        """
        data, powerflow_parameters, index = input
        if isinstance(data, HeteroData):
            n_batch = data["bus"].x.shape[0] // powerflow_parameters.n_bus
            if isinstance(self.model, HeteroSage):
                if self.mode == 'critic':
                    _, outputs, _ = self.model(data.x_dict, multipliers, data.edge_index_dict)
                elif self.mode == 'actor':
                    _, m_outputs, outputs = self.model(data.x_dict, multipliers, data.edge_index_dict, self.aux_model)
                    # self.update_eploitation_dataset(m_outputs)
                layer_outputs = OrderedDict()
                for layer, y_dict in outputs.items():
                    # reshape data to size (batch_size, n_nodes_of_type, n_features)
                    load = data["bus"].x[:, :2].view(n_batch, powerflow_parameters.n_bus, 2)
                    bus = y_dict["bus"].view(n_batch, powerflow_parameters.n_bus, 4)[..., :2]
                    gen = y_dict["gen"].view(n_batch, powerflow_parameters.n_gen, 4)[..., :2]
                    branch = y_dict["branch"].view(n_batch, powerflow_parameters.n_branch, 4)
                    V = self.parse_bus(bus)
                    Sg = self.parse_gen(gen)
                    Sd = self.parse_load(load)
                    Sf_pred, St_pred = self.parse_branch(branch)
                    if self._enforce_constraints:
                        V, Sg = self.enforce_constraints(V, Sg, powerflow_parameters)
                    layer_outputs[layer] = (pf.powerflow(V, Sd, Sg, powerflow_parameters), Sf_pred, St_pred)
                return layer_outputs, m_outputs if self.mode == 'actor' else None
            else:
                raise NotImplementedError("Unsupported model")
        elif isinstance(data, Data):
            raise NotImplementedError("Removed support for homogenous data for now.")
        else:
            raise ValueError(
                f"Unsupported data type {type(data)}, expected Data or HeteroData."
            )
        

    def enforce_constraints(
        self, V, Sg, params: pf.PowerflowParameters, strategy="sigmoid"
    ):
        """
        Ensure that voltage and power generation are within the specified bounds.

        Args:
            V: The bus voltage. Magnitude must be between params.vm_min and params.vm_max.
            Sg: The generator power. Real and reactive power must be between params.Sg_min and params.Sg_max.
            params: The powerflow parameters.
            strategy: The strategy to use for enforcing the constraints. Defaults to "sigmoid".
                "sigmoid" uses a sigmoid function to enforce the constraints.
                "clamp" uses torch.clamp to enforce the constraints.
        """
        if strategy == "sigmoid":
            fn = lambda x, lb, ub: (ub - lb) * torch.sigmoid(x) + lb
        elif strategy == "clamp":
            fn = lambda x, lb, ub: torch.clamp(x, lb, ub)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        vm = fn(V.abs(), params.vm_min, params.vm_max) if self._enforce_Vconstraints else V.abs()
        va = (V.angle() - V.angle()[:,0:1]) if self._enforce_bus_reference else V.angle()
        V = torch.polar(vm, va)
        if self._enforce_Sg_constraints:
            Sg = torch.complex(
                fn(Sg.real, params.Sg_min.real, params.Sg_max.real),
                fn(Sg.imag, params.Sg_min.imag, params.Sg_max.imag),
            )
        return V, Sg

    def supervised_loss(
        self,
        batch: PowerflowBatch,
        variables: pf.PowerflowVariables,
        Sf_pred: torch.Tensor,
        St_pred: torch.Tensor,
    ):
        """
        Calculate the MSE between the predicted and target bus voltage and generator power.
        """
        data, powerflow_parameters, _ = batch
        # parse auxiliary data from the batch
        V_target = data["bus"]["V"].view(-1, powerflow_parameters.n_bus, 2)
        Sg_target = data["gen"]["Sg"].view(-1, powerflow_parameters.n_gen, 2)
        Sf_target = data["branch"]["Sf"].view(-1, powerflow_parameters.n_branch, 2)
        St_target = data["branch"]["St"].view(-1, powerflow_parameters.n_branch, 2)
        # convert to complex numbers
        V_target = torch.complex(V_target[..., 0], V_target[..., 1])
        Sg_target = torch.complex(Sg_target[..., 0], Sg_target[..., 1])
        Sf_target = torch.complex(Sf_target[..., 0], Sf_target[..., 1])
        St_target = torch.complex(St_target[..., 0], St_target[..., 1])

        loss_voltage = (variables.V - V_target).abs().pow(2).mean()
        loss_gen = (variables.Sg - Sg_target).abs().pow(2).mean()
        loss_sf = (variables.Sf - Sf_target).abs().pow(2).mean()
        loss_st = (variables.St - St_target).abs().pow(2).mean()
        loss_power = loss_sf + loss_st
        loss_supervised = loss_voltage + loss_gen + loss_power
        self.log_dict(
            {
                "train/supervised_voltage": loss_voltage,
                "train/supervised_gen": loss_gen,
                "train/supervised_power": loss_power,
            },
            batch_size=batch.data.num_graphs,
        )
        self.log(
            "train/supervised_loss",
            loss_supervised,
            batch_size=batch.data.num_graphs,
        )
        return loss_supervised

    def powerflow_loss(
        self, batch: PowerflowBatch, variables: pf.PowerflowVariables, Sf_pred, St_pred
    ):
        """
        Loss between the predicted Sf and St by the model and the Sf and St implied by the powerflow equations.
        Assumes variables.Sf and variables.St are computed from powerflow(V).
        """
        Sf_loss = (variables.Sf - Sf_pred).abs().pow(2).mean()
        St_loss = (variables.St - St_pred).abs().pow(2).mean()
        powerflow_loss = Sf_loss + St_loss
        self.log(
            "train/powerflow_loss",
            powerflow_loss,
            batch_size=batch.data.num_graphs,
        )
        return powerflow_loss

    def _step_helper(
        self,
        variables: pf.PowerflowVariables,
        parameters: pf.PowerflowParameters,
        multipliers: Dict[str, torch.Tensor] | None = None,
        project_powermodels=False,
        output_tensors=False,
    ):
        if project_powermodels:
            variables = self.project_powermodels(variables, parameters)
        constraints, constraints_augemnted_multipliers = self.constraints(variables, parameters, multipliers)      # Constraints * multipliers
        cost, cost_tensor = self.cost(variables, parameters)                                 # OPF objective function
        if output_tensors:
            return variables, constraints_augemnted_multipliers, cost_tensor
        else:
            return variables, constraints, cost

    def on_train_epoch_start(self):
        self.multiplier_table = {
                name: [] for name in self.multiplier_metadata.keys()
            }
        # _, dual_optimizer, _ = self.optimizers()  # type: ignore
        # dual_optimizer.zero_grad()
        pass

    def on_train_epoch_end(self):
        # """
        # Where the dual ascent step happens
        # """
        # _, dual_optimizer, _ = self.optimizers()  # type: ignore
        # if self.current_epoch >= self.warmup:
        #     dual_optimizer.step()                   # Dual Ascent Step
        #     self.project_multipliers_table()        # Ensure the inequality constraints are always positive
        pass

    def training_step(self, batch: PowerflowBatch):
        primal_optimizer, dual_optimizer = self.optimizers()  # type: ignore
        # Forward Step
        multipliers = self.get_rand_multipliers(batch.index)
        layer_predictions, multipliers_predictions = self(batch, multipliers)
        if self.mode == 'actor' and self.update_common_multipliers and self.current_epoch >= 0:
            self._update_multiplier_table(multipliers_predictions)

        # evaluate training descending constraints
        constraint_layers = OrderedDict()
        for layer, (variables, Sf_pred, St_pred) in layer_predictions.items():
            if self.mode == 'critic':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
                constraint_loss = self.constraint_loss(constraints)

                powerflow_loss = self.powerflow_loss(batch, variables, Sf_pred, St_pred)
                opf_lagrangian = (
                    self.cost_weight * cost / batch.powerflow_parameters.n_gen
                    + constraint_loss
                    + self.powerflow_weight * powerflow_loss
                )
                self.log(f"{self.mode}/train/{layer}/opf_lagrangian", opf_lagrangian.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_lagrangian

            elif self.mode == 'actor':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, output_tensors=False
                )   # Evaluate the OPF objective and Constraints
                opf_violation_loss = self.constraint_loss(constraints)
                self.log(f"{self.mode}/train/{layer}/opf_violation_loss", 
                        opf_violation_loss.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/train/loss/opf_cost", 
                        cost.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_violation_loss
        training_constraints_loss = self.training_constraints(constraint_layers) 
        
        # Training objective (+/- lagrangian at the last layer)
        if self.mode == 'actor':
            multipliers = multipliers_predictions[layer]
            # Used for visualization
            opf_constraints, _ = self.constraints(variables, batch.powerflow_parameters, multipliers)
            supervised_loss = self.supervised_loss(batch, variables, Sf_pred, St_pred)

            # Computing loss
            _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
            constraint_loss = self.constraint_loss(constraints)
            opf_lagrangian = (
                self.cost_weight * cost / batch.powerflow_parameters.n_gen
                + constraint_loss
            )
            loss = - opf_lagrangian.mean()
        elif self.mode == 'critic':
            loss = opf_lagrangian.mean()
        
        # Backpropagation
        loss = loss + training_constraints_loss
        primal_optimizer.zero_grad()
        dual_optimizer.zero_grad()
        loss.backward()
        primal_optimizer.step()
        if self.current_epoch >= self.warmup:
            dual_optimizer.step()                               # Dual Ascent Step
            self.project_training_multipliers_common()          # Ensure the inequality constraints are always positive

        # Logging
        self.log(f"{self.mode}/train/loss", loss,
            prog_bar=True, batch_size=batch.data.num_graphs,
            sync_dist=True)
        
        if self.mode == 'actor':
            self.log(f"{self.mode}/train/loss/supervised", supervised_loss,
                batch_size=batch.data.num_graphs, sync_dist=True)
            for name, value in multipliers.items():
                self.log(f"{self.mode}/train/multipliers/max/{name}", value.max(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/train/multipliers/mean/{name}", value.mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                # show opf constraints
            for name, value in opf_constraints.items():
                self.log(f"{self.mode}/train/constraints/max/{name}", value['violation_max'][0].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/train/constraints/total/{name}", value['violation_mean'].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/train/constraints/mean/{name}", (value['violation_mean']/value['nConstraints']).mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/train/constraints/rate/{name}", value['rate'],
                    batch_size=batch.data.num_graphs, sync_dist=True)

        self.log(f"{self.mode}/train/loss/opf_lagrangian", opf_lagrangian.mean(),
            batch_size=batch.data.num_graphs, sync_dist=True)

        self.log(f"{self.mode}/train/loss/training_constraints", training_constraints_loss,
            batch_size=batch.data.num_graphs, sync_dist=True)

        self.log(f"{self.mode}/train/loss/training_multiplier_common/max", self.training_multipliers_common.max(),
            batch_size=batch.data.num_graphs, sync_dist=True)

        self.log(f"{self.mode}/train/loss/training_multiplier_common/mean", self.training_multipliers_common.mean(),
            batch_size=batch.data.num_graphs, sync_dist=True)


    def validation_step(self, batch: PowerflowBatch, *args):
        batch_size = batch.data.num_graphs
        multipliers = self.get_rand_multipliers(batch.index)
        layer_predictions, multipliers_predictions = self(batch, multipliers)

        constraint_layers = OrderedDict()
        for layer, (variables, Sf_pred, St_pred) in layer_predictions.items():
            if self.mode == 'critic':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
                constraint_loss = self.constraint_loss(constraints)

                powerflow_loss = self.powerflow_loss(batch, variables, Sf_pred, St_pred)
                opf_lagrangian = (
                    self.cost_weight * cost / batch.powerflow_parameters.n_gen
                    + constraint_loss
                    + self.powerflow_weight * powerflow_loss
                )
                self.log(f"{self.mode}/val/{layer}/opf_lagrangian", opf_lagrangian.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_lagrangian
            elif self.mode == 'actor':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, output_tensors=False
                )   # Evaluate the OPF objective and Constraints
                opf_violation_loss = self.constraint_loss(constraints)
                self.log(f"{self.mode}/val/{layer}/opf_violation_loss", opf_violation_loss.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_violation_loss
        val_constraints_loss = self.training_constraints(constraint_layers, train=False)
        
        # Training objective (+/- lagrangian at the last layer)
        if self.mode == 'actor':
            multipliers = multipliers_predictions[layer]
            opf_constraints, cost = self.constraints(variables, batch.powerflow_parameters, multipliers)
            supervised_loss = self.supervised_loss(batch, variables, Sf_pred, St_pred)
            _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
            constraint_loss = self.constraint_loss(constraints)
            opf_lagrangian = (
                self.cost_weight * cost / batch.powerflow_parameters.n_gen
                + constraint_loss
            )
            loss = - opf_lagrangian.mean()
            self.log(f"{self.mode}/val/loss/opf_cost", 
                        cost.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
        elif self.mode == 'critic':
            loss = opf_lagrangian.mean()
            
        loss = loss + val_constraints_loss
        self.log(f"{self.mode}/val/loss", loss,
            batch_size=batch_size, sync_dist=True)
        self.log(f"{self.mode}/val/constraints", val_constraints_loss,
            batch_size=batch_size, sync_dist=True)
    
        # if loss < self.best_val:
        #     self.best_val = loss
        #     self.log(f"{self.mode}/val/best_loss", self.best_val,
        #         batch_size=batch_size, sync_dist=True)
        if self.mode == 'actor':
            self.log(f"{self.mode}/val/loss/supervised", supervised_loss,
                batch_size=batch.data.num_graphs, sync_dist=True)
            for name, value in opf_constraints.items():
                self.log(f"{self.mode}/val/constraints/max/{name}", value['violation_max'][0].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/val/constraints/total/{name}", value['violation_mean'].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/val/constraints/mean/{name}", (value['violation_mean']/value['nConstraints']).mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/val/constraints/rate/{name}", value['rate'],
                    batch_size=batch.data.num_graphs, sync_dist=True)


    def test_step(self, batch: PowerflowBatch, *args):
        batch_size = batch.data.num_graphs
        multipliers = self.get_rand_multipliers(batch.index)
        layer_predictions, multipliers_predictions = self(batch, multipliers)

        constraint_layers = OrderedDict()
        for layer, (variables, Sf_pred, St_pred) in layer_predictions.items():
            if self.mode == 'critic':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
                constraint_loss = self.constraint_loss(constraints)

                powerflow_loss = self.powerflow_loss(batch, variables, Sf_pred, St_pred)
                opf_lagrangian = (
                    self.cost_weight * cost / batch.powerflow_parameters.n_gen
                    + constraint_loss
                    + self.powerflow_weight * powerflow_loss
                )
                self.log(f"{self.mode}/test/{layer}/opf_lagrangian", opf_lagrangian.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_lagrangian
            elif self.mode == 'actor':
                _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, output_tensors=False
                )   # Evaluate the OPF objective and Constraints
                opf_violation_loss = self.constraint_loss(constraints)
                self.log(f"{self.mode}/test/{layer}/opf_violation_loss", opf_violation_loss.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
                constraint_layers[layer] = opf_violation_loss
        val_constraints_loss = self.training_constraints(constraint_layers, train=False)
        
        # Training objective (+/- lagrangian at the last layer)
        if self.mode == 'actor':
            multipliers = multipliers_predictions[layer]
            opf_constraints, cost = self.constraints(variables, batch.powerflow_parameters, multipliers)
            supervised_loss = self.supervised_loss(batch, variables, Sf_pred, St_pred)
            _, constraints, cost = self._step_helper(
                    variables, batch.powerflow_parameters, multipliers, output_tensors=True
                )   # Evaluate the OPF objective and Constraints
            constraint_loss = self.constraint_loss(constraints)
            opf_lagrangian = (
                self.cost_weight * cost / batch.powerflow_parameters.n_gen
                + constraint_loss
            )
            loss = - opf_lagrangian.mean()
            self.log(f"{self.mode}/test/loss/opf_cost", 
                        cost.mean(),
                        batch_size=batch.data.num_graphs, sync_dist=True)
        elif self.mode == 'critic':
            loss = opf_lagrangian.mean()
            
        loss = loss + val_constraints_loss
        self.log(f"{self.mode}/test/loss", loss,
            batch_size=batch_size, sync_dist=True)
        self.log(f"{self.mode}/test/constraints", val_constraints_loss,
            batch_size=batch_size, sync_dist=True)
    
        if self.mode == 'actor':
            self.log(f"{self.mode}/test/loss/supervised", supervised_loss,
                batch_size=batch.data.num_graphs, sync_dist=True)
            for name, value in opf_constraints.items():
                self.log(f"{self.mode}/test/constraints/max/{name}", value['violation_max'][0].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/test/constraints/total/{name}", value['violation_mean'].mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/test/constraints/mean/{name}", (value['violation_mean']/value['nConstraints']).mean(),
                    batch_size=batch.data.num_graphs, sync_dist=True)
                self.log(f"{self.mode}/test/constraints/rate/{name}", value['rate'],
                    batch_size=batch.data.num_graphs, sync_dist=True)
        return loss
                
                
        # # TODO
        # # change to make faster
        # # project_powermodels taking too long
        # # go over batch w/ project pm, then individual steps without
        # multipliers = self.get_multipliers(batch.index)
        # _, constraints, cost = self._step_helper(
        #     self(batch, multipliers),
        #     batch.powerflow_parameters,
        #     project_powermodels=True,
        # )
        # test_metrics = self.metrics(cost, constraints, "test", self.detailed_metrics)
        # self.log_dict(
        #     test_metrics,
        #     batch_size=batch.data.num_graphs,
        #     sync_dist=True,
        # )
        # # TODO: rethink how to do comparison against ACOPF
        # # Test the ACOPF solution for reference.
        # # acopf_bus = self.bus_from_polar(acopf_bus)
        # # _, constraints, cost, _ = self._step_helper(
        # #     *self.parse_bus(acopf_bus),
        # #     self.parse_load(load),
        # #     project_pandapower=False,
        # # )
        # # acopf_metrics = self.metrics(
        # #     cost, constraints, "acopf", self.detailed_metrics
        # # )
        # # self.log_dict(acopf_metrics)
        # # return dict(**test_metrics, **acopf_metrics)
        # return test_metrics

    def project_powermodels(
        self,
        variables: pf.PowerflowVariables,
        parameters: pf.PowerflowParameters,
        clamp=True,
    ) -> pf.PowerflowVariables:
        V, Sg, Sd = variables.V, variables.Sg, variables.Sd
        if clamp:
            V, Sg = self.enforce_constraints(V, Sg, parameters, strategy="clamp")
        bus_shape = V.shape
        gen_shape = Sg.shape
        dtype = V.dtype
        device = V.device
        V = torch.view_as_real(V.cpu()).view(-1, parameters.n_bus, 2).numpy()
        Sg = torch.view_as_real(Sg.cpu()).view(-1, parameters.n_gen, 2).numpy()
        Sd = torch.view_as_real(Sd.cpu()).view(-1, parameters.n_bus, 2).numpy()
        # TODO: make this more robust, maybe use PyJulia
        # currently what we do is save the data to a temporary directory
        # then run the julia script and load the data back
        with TemporaryDirectory() as tempdir:
            script_path = Path(__file__).parent / "project.jl"
            busfile = Path(tempdir) / "busfile.npz"
            np.savez(busfile, V=V, Sg=Sg, Sd=Sd)
            subprocess.run(
                [
                    "julia",
                    "--project=@.",
                    script_path.as_posix(),
                    "--casefile",
                    parameters.casefile,
                    "--busfile",
                    busfile.as_posix(),
                ]
            )
            bus = np.load(busfile)
        V, Sg, Sd = bus["V"], bus["Sg"], bus["Sd"]
        # convert back to torch tensors with the original device, dtype, and shape
        V = torch.from_numpy(V)
        Sg = torch.from_numpy(Sg)
        Sd = torch.from_numpy(Sd)
        V = torch.complex(V[..., 0], V[..., 1]).to(device, dtype).view(bus_shape)
        Sg = torch.complex(Sg[..., 0], Sg[..., 1]).to(device, dtype).view(gen_shape)
        Sd = torch.complex(Sd[..., 0], Sd[..., 1]).to(device, dtype).view(bus_shape)
        return pf.powerflow(V, Sd, Sg, parameters)

    def parse_bus(self, bus: torch.Tensor):
        assert bus.shape[-1] == 2
        V = torch.complex(bus[..., 0], bus[..., 1])
        return V

    def parse_load(self, load: torch.Tensor):
        """
        Converts the load data to the format required by the powerflow module (complex tensor).

        Args:
            load: A tensor of shape (batch_size, n_features, n_bus). The first two features should contain the active and reactive load.

        """
        assert load.shape[-1] == 2
        Sd = torch.complex(load[..., 0], load[..., 1])
        return Sd

    def parse_gen(self, gen: torch.Tensor):
        assert gen.shape[-1] == 2
        Sg = torch.complex(gen[..., 0], gen[..., 1])
        return Sg

    def parse_branch(self, branch: torch.Tensor):
        assert branch.shape[-1] == 4
        Sf = torch.complex(branch[..., 0], branch[..., 1])
        St = torch.complex(branch[..., 2], branch[..., 3])
        return Sf, St

    def constraint_loss(self, constraints) -> torch.Tensor:
        """Constraints are already multiplied by Lagranian multipliers,
        return their sum
        """
        constraint_losses = []
        for val in constraints.values():
            if isinstance(val, dict):
                constraint_losses.append(val["violation_mean"])
            elif isinstance(val, torch.Tensor):
                if len(val.shape) == 1:
                    constraint_losses.append(val)
                else:
                    for i in range(val.shape[1]):
                        constraint_losses.append(val[:, i])
        if len(constraint_losses) == 0:
            return torch.zeros(1, device=self.device)
        return torch.stack(constraint_losses).sum(dim=0)

    def cost(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
    ) -> torch.Tensor:
        """Compute the cost to produce the active and reactive power."""
        p = variables.Sg.real
        p_coeff = powerflow_parameters.cost_coeff
        cost = torch.zeros_like(p)
        for i in range(p_coeff.shape[1]):
            cost += p_coeff[:, i] * p.squeeze() ** i
        # cost cannot be negative
        # cost = torch.clamp(cost, min=0)
        # # normalize the cost by the number of generators
        cost = cost.sum(dim=1) / powerflow_parameters.reference_cost
        return cost.mean(), cost

    def constraints(
        self,
        variables: pf.PowerflowVariables,
        powerflow_parameters: pf.PowerflowParameters,
        multipliers: Dict[str, torch.Tensor] | None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculates the powerflow constraints.
        :returns: Nested map from constraint name => (value name => tensor value)
        """
        constraints = pf.build_constraints(variables, powerflow_parameters)
        values = {}
        loss = {}
        for name, constraint in constraints.items():
            if isinstance(constraint, pf.EqualityConstraint):
                values[name], loss[name] = equality(
                    constraint.value,
                    constraint.target,
                    multipliers[name] if multipliers is not None else None,
                    constraint.mask,
                    self.eps,
                    constraint.isAngle,
                    self.augmented_weight if constraint.augmented else 0.0,
                )
            elif isinstance(constraint, pf.InequalityConstraint):
                values[name], loss[name] = inequality(
                    constraint.variable,
                    constraint.min,
                    constraint.max,
                    multipliers[name][..., 0] if multipliers is not None else None,
                    multipliers[name][..., 1] if multipliers is not None else None,
                    self.eps,
                    constraint.isAngle,
                    self.augmented_weight if constraint.augmented else 0.0,
                )
        return values, loss

    def training_constraints(self, opf_lagrangian_layers, train=True):
        """
        Calculate the constraint loss for the training step.
        """
        # if self.mode == 'critic':
        opf_lagrangian_tensor = torch.stack(list(opf_lagrangian_layers.values()))
        constraint_losses = opf_lagrangian_tensor[1:] - (1-self.constraint_eps) * opf_lagrangian_tensor[:-1]

        if train:
            constraint_losses = constraint_losses.mean(dim=1) * self.training_multipliers_common
        else:
            constraint_losses = constraint_losses.relu().pow(2).mean(dim=1)
        return constraint_losses.sum()
        # elif self.mode == 'actor':
        #     raise NotImplementedError("Actor mode not implemented yet.")


    def metrics(self, cost, constraints, prefix, detailed=False, train=False):
        """
        Args:
            cost: The cost of the powerflow.
            constraints: The constraints of the powerflow.
            prefix: The prefix to use for the metric names.
            detailed: Whether to log detailed
            train: Whether the metrics are for training or validation/test.
        """

        aggregate_metrics = {
            f"{prefix}/cost": [cost],
            f"{prefix}/equality/loss": [],
            f"{prefix}/equality/rate": [],
            f"{prefix}/equality/error_mean": [],
            f"{prefix}/equality/error_max": [],
            f"{prefix}/inequality/loss": [],
            f"{prefix}/inequality/rate": [],
            f"{prefix}/inequality/error_mean": [],
            f"{prefix}/inequality/error_max": [],

            f"{prefix}/equality/multiplier/mean": [],
            f"{prefix}/equality/multiplier/max": [],
            f"{prefix}/equality/multiplier/min": [],
            f"{prefix}/inequality/multiplier/mean": [],
            f"{prefix}/inequality/multiplier/max": [],
            f"{prefix}/inequality/multiplier/min": [],
        }


        detailed_metrics = {}
        reduce_fn = {
            "default": torch.sum,
            "error_mean": torch.mean,
            "error_max": torch.max,
            "rate": torch.mean,
            "multiplier/mean": torch.mean,
            "multiplier/max": torch.max,
        }

        for constraint_name, constraint_values in constraints.items():
            constraint_type = constraint_name.split("/")[0]
            for value_name, value in constraint_values.items():
                if detailed:
                    detailed_metrics[f"{prefix}/{constraint_name}/{value_name}"] = value
                aggregate_name = f"{prefix}/{constraint_type}/{value_name}"
                aggregate_metrics[aggregate_name].append(value.reshape(1))
        for aggregate_name in aggregate_metrics:
            value_name = aggregate_name.rsplit("/", 1)[1]
            fn = (
                reduce_fn[value_name]
                if value_name in reduce_fn
                else reduce_fn["default"]
            )
            aggregate_metrics[aggregate_name] = fn(
                torch.stack(aggregate_metrics[aggregate_name])
            )
        return {**aggregate_metrics, **detailed_metrics}

    def configure_optimizers(self):
        primal_optimizer = torch.optim.AdamW(  # type: ignore
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            fused=True,
        )
        # dual_optimizer = torch.optim.AdamW(  # type: ignore
        #     self.multipliers_table.parameters(),
        #     lr=self.lr_dual,
        #     weight_decay=self.weight_decay_dual,
        #     maximize=True,
        #     fused=True,
        # )
        dual_optimizer = torch.optim.AdamW(  # type: ignore
            [self.training_multipliers_common],
            lr=self.lr_common,
            weight_decay=self.weight_decay_dual,
            fused=True,
            maximize=True,
        )

        optimizers = [primal_optimizer, dual_optimizer]
        return optimizers

    @staticmethod
    def bus_from_polar(bus):
        """
        Convert bus voltage from polar to rectangular.
        """
        bus = bus.clone()
        V = torch.polar(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.real
        bus[:, 1, :] = V.imag
        return bus

    @staticmethod
    def bus_to_polar(bus):
        """
        Convert bus voltage from rectangular to polar.
        """
        bus = bus.clone()
        V = torch.complex(bus[:, 0, :], bus[:, 1, :])
        bus[:, 0, :] = V.abs()
        bus[:, 1, :] = V.angle()
        return bus