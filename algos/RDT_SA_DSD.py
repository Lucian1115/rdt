import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, Tuple
import os
import json
import traceback
import gym
import numpy as np
import pyrallis
import torch
from dataclasses import dataclass
from tqdm.auto import trange

import utils.functions as func
import utils.dt_functions as dt_func
from utils.logger import init_logger, Logger
from utils.attack import Evaluation_Attacker
from utils.run_mean_std import RunningMeanStd

# 引入新模块
from utils.robust_dt_model import RobustDecisionTransformer
from utils.robust_eval import eval_fn_robust
from algos.RDT import loss_fn, correct_outliers # 复用原有的辅助函数

MODEL_PATH = {
    "IQL": os.path.join(os.path.dirname(os.path.dirname(__file__)), "IQL_model"),
}

@dataclass
class TrainConfig:
    # --- SA-DSD 特有配置 ---
    use_fdep: bool = True           # 开启频域嵌入纯化器 (创新点一)
    use_ttsc: bool = True           # 开启测试时自洽性提示微调 (创新点二)
    ttsc_steps: int = 3             # TTSC 优化步数
    ttsc_lr: float = 0.01           # TTSC 学习率
    # ---------------------
    
    # Experiment Defaults
    n_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 100
    num_updates_on_epoch: int = 1000
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1 
    mlp_embedding: bool = False
    mlp_head: bool = False
    mlp_reward: bool = True
    embed_order: str = "rsa"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    reward_scale: float = 0.001
    normalize: bool = True
    normalize_reward: bool = False
    loss_fn: str = "wmse"
    wmse_coef: float = (1.0, 1.0)
    reward_coef: float = 1.0
    recalculate_return: bool = False
    correct_freq: int = 50
    correct_start: int = 50
    correct_thershold: Tuple[float] = (6.0, 6.0)
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_id: str = "00"
    eval_only: bool = False
    eval_attack: bool = True
    checkpoint_dir: str = None
    test_time: str = None
    use_wandb: int = 0
    group: str = "RDT-SA-DSD"
    env: str = "walker2d-medium-replay-v2"
    seed: int = 0
    down_sample: bool = True
    sample_ratio: float = 0.1
    debug: bool = False
    alg_type: str = "RDT-SA-DSD"
    logdir: str = "results"
    dataset_path: str = "/home/user/.d4rl/datasets"
    save_model: bool = True
    debug_eval: bool = False
    corruption_agent: str = "IQL"
    corruption_seed: int = 0
    corruption_mode: str = "random" 
    corruption_tag: str = "act"
    corruption_obs: float = 0.0
    corruption_act: float = 1.0
    corruption_rew: float = 0.0
    corruption_rate: float = 0.3
    use_original: int = 0
    same_index: int = 0
    froce_attack: int = 0

    def __post_init__(self):
        if not self.eval_only:
            if self.corruption_tag == "obs":
                self.corruption_obs = 1.0
                self.corruption_act = 0.0
                self.corruption_rew = 0.0
            if self.corruption_tag == "act":
                self.corruption_obs = 0.0
                self.corruption_act = 1.0
                self.corruption_rew = 0.0
            if self.corruption_tag == "rew":
                self.corruption_obs = 0.0
                self.corruption_act = 0.0
                self.corruption_rew = 1.0
            
            if self.env.startswith("walker"):
                self.target_returns = [5000]
                self.reward_scale = 0.001

            self.eval_every = int(self.num_epochs / 10)
            self.update_steps = int(self.num_epochs * self.num_updates_on_epoch)
            self.warmup_steps = int(0.1 * self.update_steps)

        self.eval_attack_mode = self.corruption_mode
        self.eval_attack_eps = 0.5
        self.eval_corruption_rate = 1.0

def set_model(config: TrainConfig):
    model = RobustDecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        mlp_embedding=config.mlp_embedding,
        mlp_head=config.mlp_head,
        mlp_reward=config.mlp_reward,
        embed_order=config.embed_order,
        use_fdep=config.use_fdep, # 传入配置
    ).to(config.device)
    return model

def compute_loss(config, model, batch):
    log_dict, debug_dict = {}, {}
    states, actions, returns, rewards, time_steps, mask, attack_mask, traj_indexs = [b.to(config.device) for b in batch]
    padding_mask = ~mask.to(torch.bool)

    predicted = model(
        states=states,
        actions=actions,
        returns_to_go=returns,
        time_steps=time_steps,
        padding_mask=padding_mask,
        apply_fdep=True # 训练时始终启用 FDEP
    )
    predicted_actions, predicted_rewards = predicted

    loss = loss_fn(config, predicted_actions, actions, mask, config.wmse_coef[0])
    log_dict.update({"loss_action": loss.item()})
    
    loss_reward = loss_fn(config, predicted_rewards, rewards, mask, config.wmse_coef[1])
    loss += config.reward_coef * loss_reward
    log_dict.update({"loss_reward": loss_reward.item()})
    log_dict.update({"policy_loss": loss.item()})
    
    data_info = None
    if config.correct_thershold is not None:
        data_info = {
            "actions": [predicted_actions, actions], "rewards": [predicted_rewards, rewards],
            "mask": mask,  "attack_mask": attack_mask,
            "traj_indexs": traj_indexs, "time_steps": time_steps,
        }
    return loss, log_dict, debug_dict, data_info

def train(config: TrainConfig, logger: Logger):
    func.set_seed(config.seed)
    env = gym.make(config.env)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.shape[0]
    config.max_action = float(env.action_space.high[0])
    
    dataset = dt_func.SequenceDataset(config, logger)
    logger.info(f"Dataset size: {len(dataset.dataset)}")
    env = func.wrap_env(env, state_mean=dataset.state_mean, state_std=dataset.state_std, reward_scale=config.reward_scale)
    env.seed(config.seed)

    model = set_model(config)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda steps: min((steps + 1) / config.warmup_steps, 1))

    data_dist = []
    if config.correct_thershold is not None:
        for thershold in config.correct_thershold:
             data_dist.append(RunningMeanStd(thershold=thershold) if thershold > 0.0 else None)

    if config.eval_attack:
        state_std, act_std, rew_std, rew_min = func.get_state_std(config)
        eval_attacker = Evaluation_Attacker(
            config, config.env, config.corruption_agent, config.eval_attack_eps,
            config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
            MODEL_PATH[config.corruption_agent],
        )
    else:
        eval_attacker = None

    total_updates = 0
    for epoch in trange(1, config.num_epochs + 1, desc="Training"):
        for step in trange(config.num_updates_on_epoch, desc="Step", leave=False):
            batch = dataset.get_batch(config.batch_size, config.recalculate_return)
            config.recalculate_return = False
            optim.zero_grad()
            loss, log_dict, debug_dict, data_info = compute_loss(config, model, batch)
            loss.backward()
            if config.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optim.step()
            scheduler.step()
            total_updates += 1
            
            if config.correct_thershold is not None:
                correct = epoch > config.correct_start and step % config.correct_freq == 0
                correct_info, correct_dict = correct_outliers(config, data_info, data_dist, correct=correct)
                if correct:
                    for name, info in correct_info.items():
                        dataset.correct(info["traj_indexs"], info["time_steps"], info["correct_data"], name)
                    if config.correct_thershold is not None and config.correct_thershold[1] > 0.0:
                        config.recalculate_return = True

        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.num_epochs:
            eval_log = eval_fn_robust(config, env, model, eval_attacker)
            logger.record("epoch", epoch)
            for k, v in eval_log.items(): logger.record(k, v)
            for k, v in log_dict.items(): logger.record(f"update/{k}", v)
            logger.dump(epoch)
            
            if config.save_model and epoch == config.num_epochs:
                torch.save(model.state_dict(), os.path.join(logger.get_dir(), f"final_policy_sa_dsd.pth"))

def test(config: TrainConfig, logger: Logger):
    func.set_seed(config.seed)
    env = gym.make(config.env)
    dataset = dt_func.SequenceDataset(config, logger)
    env = func.wrap_env(env, state_mean=dataset.state_mean, state_std=dataset.state_std, reward_scale=config.reward_scale)
    model = set_model(config)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, "final_policy_sa_dsd.pth")))
    model.eval()
    
    if config.eval_attack:
        state_std, act_std, rew_std, rew_min = func.get_state_std(config)
        eval_attacker = Evaluation_Attacker(
            config, config.env, config.corruption_agent, config.eval_attack_eps,
            config.state_dim, config.action_dim, state_std, act_std, rew_std, rew_min, config.eval_attack_mode,
            MODEL_PATH[config.corruption_agent],
        )
    else:
        eval_attacker = None

    eval_log = eval_fn_robust(config, env, model, eval_attacker)
    for k, v in eval_log.items(): logger.record(k, v)
    logger.dump(0)

@pyrallis.wrap()
def main(config: TrainConfig):
    logger = init_logger(config)
    try:
        if config.eval_only: test(config, logger)
        else: train(config, logger)
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")

if __name__ == "__main__":
    main()
