import argparse
import os
import typing
import copy
from functools import partial
from pathlib import Path

import optuna
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from wandb.wandb_run import Run

from src.opf.dataset import CaseDataModule
from src.opf.hetero import HeteroGCN, HeteroSage, critic_heteroSage, actor_heteroSage
from src.opf.modules import OPFDual
from src.opf.unrolling import OPFUnrolled


def main():
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--operation", default="train", choices=["train", "study"])
    parser.add_argument("--run_name", type=str, default="actor_training")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--no_log", action="store_false", dest="log")
    parser.add_argument(
        "--hotstart", type=str, default=None, help="ID of run to hotstart with."
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--no_compile", action="store_false", dest="compile", default=False
    )   # use torch compile
    parser.add_argument(
        "--no_personalize", action="store_false", dest="personalize", default=False
    )   # 
    
    # Solver arguments
    parser.add_argument("--solver", type=str, choices=["unrolling", "primal_dual"], default="unrolling")
    parser.add_argument('--mode', type=str, choices=['critic', 'actor', 'actor-critic', None], default='actor-critic')
    parser.add_argument('--num_cycles', type=int, default=30)
    

    # data arguments
    parser.add_argument("--case_name", type=str, default="case30_ieee")#default="case179_goc__api")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fast_dev_run", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--homo", action="store_true", default=False)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--no_gpu", action="store_false", dest="gpu")
    group.add_argument("--max_epochs", type=int, default=3)
    group.add_argument("--max_epochs_critic", type=int, default=5)
    group.add_argument("--max_epochs_actor", type=int, default=5)
    group.add_argument("--patience", type=int, default=15)
    group.add_argument("--gradient_clip_val", type=float, default=0)

    # the gnn being used
    critic_heteroSage.add_args(parser)
    actor_heteroSage.add_args(parser)
    OPFUnrolled.add_args(parser)

    params = parser.parse_args()
    params_dict = vars(params)

    torch.set_float32_matmul_precision("high")
    if params.operation == "study":
        study(params_dict)
    elif params.operation == "train":
        if params_dict["mode"] in ["critic", "actor"]:
            trainer = make_trainer(params_dict, params["mode"])
            train(trainer, params_dict)
        elif params_dict["mode"] == "actor-critic":
            critic_trainer = make_trainer(params_dict, "critic", callbacks=[])
            actor_trainer = make_trainer(params_dict, "actor", callbacks=[])
            train((critic_trainer, actor_trainer), params_dict)
    else:
        raise ValueError(f"Unknown operation: {params.operation}")

def reset_trainer(trainer):
    trainer.fit_loop.epoch_progress.current.started = 0
    trainer.fit_loop.epoch_progress.current.processed = 0
    trainer.fit_loop.epoch_progress.current.completed = 0
    trainer.fit_loop.epoch_progress.current.ready = 0

def make_trainer(params, mode, callbacks=[], wandb_kwargs={}):
    logger = None
    if params["log"]:
        logger = WandbLogger(
            project="opf",
            save_dir=params["log_dir"],
            config=params,
            log_model=True,
            notes=params["notes"],
            **wandb_kwargs,
        )

        logger.log_hyperparams(params)
        typing.cast(Run, logger.experiment).log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                (path.endswith(".py") or path.endswith(".jl"))
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )

        # logger specific callbacks
        callbacks += [
            ModelCheckpoint(
                monitor=f"{mode}/val/loss",
                dirpath=Path(params["log_dir"]) / "checkpoints" / Path(params["mode"]) / Path(logger.experiment.id) ,
                filename=f"best_{mode}",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=1,
            )
        ]
    # callbacks += [EarlyStopping(monitor=f"{mode}/val/loss", patience=params["patience"])]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=32,
        accelerator="cuda" if params["gpu"] else "cpu",
        devices=1,
        max_epochs=params[f"max_epochs_{mode}"],
        default_root_dir=params["log_dir"],
        fast_dev_run=params["fast_dev_run"],
        gradient_clip_val=params["gradient_clip_val"],
        log_every_n_steps=1,
    )
    return trainer


def _train(trainer: Trainer, params):
    dm = CaseDataModule(pin_memory=params["gpu"], **params)
    solver = params['solver']
    
    # initialize NN optimizers
    if solver == 'primal_dual':
        if params["homo"]:
            # gcn = GCN(in_channels=dm.feature_dims, out_channels=4, **params)
            raise NotImplementedError("Homogenous model not currently implemented.")
        else:
            dm.setup()
            gcn = HeteroSage(
                dm.metadata(),
                in_channels=max(dm.feature_dims.values()),
                out_channels=4,
                **params,
            )
            gcn = typing.cast(
                HeteroSage,
                torch.compile(
                    gcn.cuda(), dynamic=False, fullgraph=True, disable=not params["compile"]
                ),
            )
    elif solver == 'unrolling':
        dm.setup()
        gcn_critic = critic_heteroSage(
            dm.metadata(),
            x_channels=max(dm.feature_dims.values()),
            lambda_channels=6,
            out_channels=4,
            **params,
        )
        gcn_critic = typing.cast(
            critic_heteroSage,
            torch.compile(
                gcn_critic.cuda(), dynamic=False, fullgraph=True, disable=not params["compile"]
            ),
        )
        gcn_actor = actor_heteroSage(
            dm.metadata(),
            x_channels=max(dm.feature_dims.values()),
            lambda_channels=6,
            y_channels=4,
            **params,
        )
        gcn_actor = typing.cast(
            actor_heteroSage,
            torch.compile(
                gcn_actor.cuda(), dynamic=False, fullgraph=True, disable=not params["compile"]
            ),
        )
    else:
        raise NotImplementedError

    assert dm.powerflow_parameters is not None
    n_nodes = (
        dm.powerflow_parameters.n_bus,
        dm.powerflow_parameters.n_branch,
        dm.powerflow_parameters.n_gen,
    )

    # Training
    if solver == 'unrolling':
        if params['mode'] == 'critic':
            model = OPFUnrolled(
                gcn_critic, n_nodes, multiplier_table_length=0, **params  # type: ignore
            )
            trainer.fit(model, dm)
        elif params['mode'] == 'actor':
            # load the critic
            checkpoint = f'logs/checkpoints/critic/3lheyqhg/best.ckpt'
            critic_model = OPFUnrolled.load_from_checkpoint(checkpoint, model=gcn_critic)
            model = OPFUnrolled(
                gcn_actor, n_nodes, aux_model=critic_model.model,
                multiplier_table_length=0, **params
                )        
            trainer.fit(model, dm)
        elif params['mode'] == 'actor-critic':
            critic_trainer, actor_trainer = trainer
            critic_model = OPFUnrolled(
                gcn_critic, n_nodes, multiplier_table_length=2e6, mode='critic',
                **{k: v for k, v in params.items() if k != 'mode'})
            
            actor_model = OPFUnrolled(
                gcn_actor, n_nodes, aux_model=copy.deepcopy(gcn_critic),
                multiplier_table_length=2e6, mode='actor',
                **{k: v for k, v in params.items() if k != 'mode'})

            best_critic_loss = torch.inf
            best_actor_loss = torch.inf
            for cycle in range(params['num_cycles']):
                critic_model._generate_exploitation_dataset(actor_model.multiplier_table, n_samples=50000)
                critic_trainer.fit(critic_model, dm)
                checkpoint = f'logs/checkpoints/actor-critic/{critic_trainer.logger.experiment.id}/best_critic.ckpt'
                critic_model = OPFUnrolled.load_from_checkpoint(checkpoint, model=gcn_critic)
                if critic_trainer.callbacks[2].kth_value < best_critic_loss:
                    best_critic_loss = critic_trainer.callbacks[2].kth_value

                actor_model.aux_model = copy.deepcopy(critic_model.model).eval()
                actor_trainer.fit(actor_model, dm)
                # checkpoint = f'logs/checkpoints/actor-critic/{actor_trainer.logger.experiment.id}/best_actor.ckpt'
                # actor_model = OPFUnrolled.load_from_checkpoint(checkpoint, model=gcn_actor)
                if actor_trainer.callbacks[2].kth_value < best_actor_loss:
                    best_actor_loss = actor_trainer.callbacks[2].kth_value
                reset_trainer(critic_trainer), reset_trainer(actor_trainer)

                # logging beast losses
                critic_trainer.logger.log_metrics({'best_critic_loss': best_critic_loss})
                actor_trainer.logger.log_metrics({'best_actor_loss': best_actor_loss})

                

    elif solver == 'primal_dual':
        model = OPFDual(
            gcn, n_nodes, multiplier_table_length=len(dm.train_dataset) if params["personalize"] else 0, **params  # type: ignore
        )
        trainer.fit(model, dm)


def train(trainer: Trainer, params):
    _train(trainer, params)
    # trainer.test(model, dm)
    if isinstance(trainer, tuple):
        for t in trainer:
            for logger in t.loggers:
                logger.finalize("finished")
    else:
        for logger in trainer.loggers:
            logger.finalize("finished")


def study(params: dict):
    study_name = "opf-5"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=50, max_resource=1000, reduction_factor=3
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
        directions=["minimize"],
    )
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: dict):
    params = dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-16, 1, log=True),
        lr_dual=trial.suggest_float("lr_dual", 1e-5, 1.0, log=True),
        lr_common=trial.suggest_float("lr_common", 1e-5, 1.0, log=True),
        weight_decay_dual=trial.suggest_float("weight_decay_dual", 1e-16, 1, log=True),
        dropout=trial.suggest_float("dropout", 0, 1),
        supervised_weight=100.0,
        augmented_weight=10.0,
        powerflow_weight=1.0,
        case_name="case118_ieee",
        n_layers=20,
        batch_size=100,
        n_channels=128,
        cost_weight=1.0,
        equality_weight=1e3,
        max_epochs=1000,
        patience=1000,
        warmup=10,
        supervised_warmup=20,
        # # MLP parameteers
        mlp_hidden_channels=512,
        mlp_read_layers=2,
        mlp_per_gnn_layers=2,
    )
    params = {**default_params, **params}
    trainer = make_trainer(
        params,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/invariant"
            )
        ],
        wandb_kwargs=dict(group=trial.study.study_name),
    )

    train(trainer, params)

    # finish up
    if isinstance(trainer.logger, WandbLogger):
        trial.set_user_attr("wandb_id", trainer.logger.experiment.id)
    for logger in trainer.loggers:
        logger.finalize("finished")

    print(trainer.callback_metrics)
    return trainer.callback_metrics["val/invariant"].item()


if __name__ == "__main__":
    main()