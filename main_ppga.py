import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import torch
import time
import pickle
import numpy as np

v = torch.ones(1, device='cuda')  # init torch cuda before jax

from baselines.PPGA.algorithm.config_ppga import PPGAConfig

import warnings

from baselines.PPGA.algorithm.train_ppga import train_ppga
from baselines.PPGA.envs.brax_custom.brax_env import make_vec_env_brax_ppga
from baselines.PPGA.utils.utilities import config_wandb, log
from qdax.utils.metrics import CSVLogger

import hydra
import jax
import jax.numpy as jnp

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


@hydra.main(version_base="1.2", config_path="configs/", config_name="ppga")
def main(hydra_config):
    # Verify config
    cfg = PPGAConfig.create(hydra_config)
    cfg = cfg.as_dot_dict()

    cfg.num_emitters = 1

    vec_env = make_vec_env_brax_ppga(task_name=hydra_config.env.name, batch_size=cfg.env_batch_size,
                                     seed=cfg.seed, backend=hydra_config.env.backend, clip_obs_rew=cfg.clip_obs_rew, episode_length=hydra_config.env.episode_length)
    

    vec_env_eval = make_vec_env_brax_ppga(task_name=hydra_config.env.name, batch_size=3000,
                                    seed=cfg.seed, backend=hydra_config.env.backend, clip_obs_rew=cfg.clip_obs_rew, episode_length=hydra_config.env.episode_length)

    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    jax.debug.print("SHAPE: {}", cfg.obs_shape)
    cfg.action_shape = vec_env.single_action_space.shape

    cfg.bd_min = vec_env.behavior_descriptor_limits[0][0]
    cfg.bd_max = vec_env.behavior_descriptor_limits[1][0]

    # Setup metrics dictionary
    metrics = dict.fromkeys(
        [
            "iteration", 
            "qd_score", 
            "coverage", 
            "max_fitness", 
            "time", 
            "evaluation",
        ], 
        jnp.array([])
    )
    
    # Setup CSV logger
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )
    
    # Setup directory for metrics
    metrics_file_path = "./metrics_incremental.pickle"
    
    # Disable wandb
    cfg.use_wandb = False
    
    outdir = os.path.join(cfg.expdir, str(cfg.seed))
    cfg.outdir = outdir
    assert not os.path.exists(outdir) or cfg.load_scheduler_from_cp is not None or cfg.load_archive_from_cp is not None, \
        f"Warning: experiment dir {outdir} exists. Danger of overwriting previous run"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not cfg.save_scheduler:
        log.warning('Warning. You have set save scheduler to false. Only the archive dataframe will be saved in each '
                    'checkpoint. If you plan to restart this experiment from a checkpoint or wish to have the added '
                    'safety of recovering from a potential crash, it is recommended that you enable save_scheduler.')

    # Run the training, passing the logger and metrics path
    train_ppga(cfg, vec_env, vec_env_eval, csv_logger=csv_logger, metrics_file_path=metrics_file_path)


if __name__ == "__main__":
    main()
