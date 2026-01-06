import torch
import torch.nn as nn
import numpy as np

def select_action_with_ttsc(
    model, 
    states, 
    actions, 
    returns, 
    time_steps, 
    steps=3, 
    lr=1e-2,
    device='cuda'
):
    """
    测试时自洽性提示微调 (TTSC-PT) - 创新点二
    
    Args:
        model: RobustDecisionTransformer 实例
        states: 当前状态序列 [1, seq_len, state_dim]
        actions, returns, time_steps: 其他上下文
        steps: 优化步数
        lr: 学习率
    Returns:
        optimized_action: 微调后的动作
    """
    # 确保模型处于 eval 模式 (BN层冻结等)
    model.eval() 
    
    # 1. 初始化可学习的 Prompt (Delta)，直接加在 State 上
    # shape 与 states 一致, requires_grad=True 允许求导
    delta = torch.zeros_like(states, requires_grad=True, device=device)
    
    # 优化器只优化 delta，不优化 model 参数
    optimizer = torch.optim.SGD([delta], lr=lr)
    
    # 2. 内循环优化 (Test-Time Loop)
    for _ in range(steps):
        optimizer.zero_grad()
        
        # 加上扰动后的状态输入
        perturbed_states = states + delta
        
        # --- 构建视图 (View Construction) ---
        # 视图 A (Standard View): 不经过 FDEP 处理
        # 这代表模型在"未增强"视角下的预测
        pred_actions_v1, _ = model(
            perturbed_states, actions, returns, time_steps, 
            apply_fdep=False
        )
        
        # 视图 B (Augmented View): 经过 FDEP 处理
        # 这代表模型在"频域纯化"视角下的预测
        # 如果模型对两者输出一致，说明输入对于频域扰动鲁棒
        pred_actions_v2, _ = model(
            perturbed_states, actions, returns, time_steps, 
            apply_fdep=True
        )
        
        # --- 自洽性损失 (Consistency Loss) ---
        # 最小化两个视图输出的差异，抵抗 Action Diff 攻击
        # 只取最后一个时间步(当前步)的动作差异
        loss = torch.norm(pred_actions_v1[:, -1] - pred_actions_v2[:, -1], p=2)
        
        # 反向传播: 计算 loss 对 delta 的梯度
        loss.backward()
        
        # 更新 delta
        optimizer.step()
        
    # 3. 最终推断
    # 使用优化后的 delta 和 FDEP 开启的状态进行最终预测
    # 此时不需要梯度
    with torch.no_grad():
        final_states = states + delta
        final_actions_pred, _ = model(
            final_states, actions, returns, time_steps, 
            apply_fdep=True # 最终推断使用纯化后的特征
        )
        
    # 返回最后一个时间步的动作 [action_dim]
    return final_actions_pred[0, -1].cpu().numpy()
