import torch
import numpy as np
import gym
from tqdm.auto import trange
from utils.ttsc_pt import select_action_with_ttsc

@torch.no_grad()
def eval_rollout_robust(
    model,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
    eval_attacker = None,
    eval_corruption_rate: float = 0.0,
    eval_attack_tag: str = "obs",
    use_ttsc: bool = False,
    ttsc_steps: int = 3,
    ttsc_lr: float = 0.01
):
    model.eval()
    
    state_dim = model.state_dim
    action_dim = model.action_dim
    episode_len = model.episode_len
    seq_len = model.seq_len
    
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]

    states = torch.zeros(
        1, episode_len + 1, state_dim, dtype=torch.float32, device=device
    )
    actions = torch.zeros(
        1, episode_len, action_dim, dtype=torch.float32, device=device
    )
    returns = torch.zeros(
        1, episode_len + 1, 1, dtype=torch.float32, device=device
    )
    time_steps = torch.arange(episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    obs = env.reset()
    
    # Initial Observation Attack
    if eval_attacker is not None and eval_attack_tag == "obs":
        if np.random.rand() < eval_corruption_rate:
            obs = eval_attacker.attack_obs(obs)
            
    states[:, 0] = torch.as_tensor(obs, device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    episode_return, episode_len_count = 0.0, 0.0

    for step in range(episode_len):
        # Context window
        ctx_start = max(0, step - seq_len + 1)
        ctx_end = step + 1
        
        cur_states = states[:, :ctx_end][:, -seq_len:]
        cur_actions = actions[:, :ctx_end][:, -seq_len:]
        cur_returns = returns[:, :ctx_end][:, -seq_len:]
        cur_timesteps = time_steps[:, :ctx_end][:, -seq_len:]

        # Padding
        if cur_states.shape[1] < seq_len:
            pad_len = seq_len - cur_states.shape[1]
            cur_states = torch.cat([torch.zeros(1, pad_len, state_dim, device=device), cur_states], dim=1)
            cur_actions = torch.cat([torch.zeros(1, pad_len, action_dim, device=device), cur_actions], dim=1)
            cur_returns = torch.cat([torch.zeros(1, pad_len, 1, device=device), cur_returns], dim=1)
            cur_timesteps = torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device=device), cur_timesteps], dim=1)

        # Action Selection
        if use_ttsc:
            # 开启梯度计算以进行 TTSC
            with torch.enable_grad():
                predicted_action = select_action_with_ttsc(
                    model, 
                    cur_states, 
                    cur_actions, 
                    cur_returns, 
                    cur_timesteps,
                    steps=ttsc_steps,
                    lr=ttsc_lr,
                    device=device
                )
        else:
            predicted_act_seq, _ = model(cur_states, cur_actions, cur_returns, cur_timesteps)
            predicted_action = predicted_act_seq[0, -1].cpu().numpy()

        # Action Attack (Test-time Adversarial Attack)
        if eval_attacker is not None and eval_attack_tag == "act":
            if np.random.rand() < eval_corruption_rate:
                predicted_action = eval_attacker.attack_act(predicted_action)

        predicted_action = np.clip(predicted_action, *action_range)
        next_state, reward, done, info = env.step(predicted_action)

        episode_return += reward
        episode_len_count += 1

        # Next Observation Attack
        if eval_attacker is not None and eval_attack_tag == "obs":
            if np.random.rand() < eval_corruption_rate:
                next_state = eval_attacker.attack_obs(next_state)
        
        # Reward Attack
        if eval_attacker is not None and eval_attack_tag == "rew":
             if np.random.rand() < eval_corruption_rate:
                 reward = eval_attacker.attack_rew(reward)

        actions[:, step] = torch.as_tensor(predicted_action, device=device)
        if step < episode_len - 1:
            states[:, step + 1] = torch.as_tensor(next_state, device=device)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward, device=device)

        if done:
            break

    return episode_return, episode_len_count

def eval_fn_robust(config, env, model, eval_attacker=None):
    eval_log = {}
    eval_attack_tag = "obs" # default
    if config.corruption_obs > 0: eval_attack_tag = "obs"
    if config.corruption_act > 0: eval_attack_tag = "act"
    if config.corruption_rew > 0: eval_attack_tag = "rew"
    
    if config.eval_attack and config.corruption_tag == "":
         eval_attack_tag = "obs"

    use_ttsc = getattr(config, 'use_ttsc', False)
    ttsc_steps = getattr(config, 'ttsc_steps', 3)
    ttsc_lr = getattr(config, 'ttsc_lr', 1e-2)

    for target_return in config.target_returns:
        eval_returns = []
        for _ in trange(config.n_episodes, desc="Robust Evaluation", leave=False):
            eval_return, eval_len = eval_rollout_robust(
                model=model,
                env=env,
                target_return=target_return * config.reward_scale,
                eval_attacker=eval_attacker,
                eval_corruption_rate=config.eval_corruption_rate,
                eval_attack_tag=eval_attack_tag,
                device=config.device,
                use_ttsc=use_ttsc,
                ttsc_steps=ttsc_steps,
                ttsc_lr=ttsc_lr
            )
            eval_returns.append(eval_return / config.reward_scale)

        eval_returns = np.array(eval_returns)
        normalized_score = env.get_normalized_score(eval_returns) * 100
        eval_log.update({
            f"eval/{target_return}_normalized_score_mean": np.mean(normalized_score),
        })
    return eval_log
