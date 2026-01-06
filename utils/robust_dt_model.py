import torch
import torch.nn as nn
# 假设 utils.dt_functions 和 utils.networks 在路径中可用
from utils.dt_functions import DecisionTransformer
from utils.fdep import FrequencyDomainPurifier

class RobustDecisionTransformer(DecisionTransformer):
    """
    集成 FDEP 的鲁棒决策 Transformer
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 20,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        predict_dropout: float = 0.1,
        mlp_embedding: bool = False,
        mlp_head: bool = False,
        mlp_reward: bool = False,
        embed_order: str = "rsa",
        use_fdep: bool = True, # 开关
    ):
        # 初始化父类
        super().__init__(
            state_dim, action_dim, seq_len, episode_len, embedding_dim,
            num_layers, num_heads, attention_dropout, residual_dropout,
            embedding_dropout, predict_dropout, mlp_embedding, mlp_head,
            mlp_reward, True, embed_order # predict_reward=True
        )
        
        self.use_fdep = use_fdep
        if self.use_fdep:
            # FDEP 作用于序列长度维度
            self.fdep = FrequencyDomainPurifier(embedding_dim, seq_len)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: torch.Tensor = None,
        apply_fdep: bool = True, # 动态控制 FDEP (用于 TTSC)
    ):
        if hasattr(self, "state_dropout"):
            states = self.state_dropout(states)
            
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embeddings
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states)
        act_emb = self.action_emb(actions)
        returns_emb = self.return_emb(returns_to_go)

        # --- 创新点一: FDEP 介入 ---
        # 仅对 State Embedding 进行纯化，应对观测噪声
        if self.use_fdep and apply_fdep:
            state_emb = self.fdep(state_emb)
        # -------------------------

        # 堆叠序列 [batch_size, seq_len * 3, emb_dim]
        if self.embed_order == "rsa":
            sequence = torch.stack([returns_emb, state_emb, act_emb], dim=1)
        elif self.embed_order == "sar":
            sequence = torch.stack([state_emb, act_emb, returns_emb], dim=1)
        else:
            raise ValueError(f"Invalid embedding order {self.embed_order}.")
            
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        sequence = sequence + time_emb.repeat_interleave(3, dim=1)
        
        # 处理 padding mask
        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )

        out = self.emb_norm(sequence)
        
        if hasattr(self, "emb_drop"): 
            out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out = self.predict_dropout(out)

        # 拆分输出
        if self.embed_order == "rsa":
            out_r_emb, out_s_emb, out_a_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]
        elif self.embed_order == "sar":
            out_s_emb, out_a_emb, out_r_emb = out[:, 0::3], out[:, 1::3], out[:, 2::3]

        action_out = self.action_head(out_s_emb)
        
        reward_out = None
        if self.predict_reward:
            reward_out = self.reward_head(out_a_emb)

        return action_out, reward_out
