import torch
import torch.nn as nn
import torch.fft
import math

class FrequencyDomainPurifier(nn.Module):
    """
    频域嵌入纯化器 (FDEP) - 创新点一
    利用可学习的频谱滤波器在特征层自适应地分离信号与噪声。
    """
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        
        # rfft 输出的频率分量数量
        self.freq_len = seq_len // 2 + 1
        
        # 初始化可学习的复数权重 (Complex Weight)
        # 形状: [freq_len, embed_dim, 2] (最后维度2代表实部和虚部)
        self.complex_weight = nn.Parameter(
            torch.zeros(self.freq_len, embed_dim, 2, dtype=torch.float32)
        )
        
        # --- 初始化策略: 高斯衰减 (模拟低通滤波器) ---
        # 我们希望初始状态下，低频部分保留，高频部分抑制
        # 频率索引: 0 (直流) -> freq_len-1 (最高频)
        with torch.no_grad():
            # 生成频率坐标 [0, 1, ..., freq_len-1]
            freq_indices = torch.arange(self.freq_len, dtype=torch.float32).unsqueeze(1) # [freq_len, 1]
            
            # 高斯衰减系数: exp(-f^2 / (2 * sigma^2))
            # sigma 越大，通带越宽。这里设为 freq_len 的一半，提供温和的低通效果
            sigma = self.freq_len / 2.0
            decay = torch.exp(-(freq_indices**2) / (2 * sigma**2))
            
            # 将衰减系数应用到实部，虚部初始化为微小随机噪声
            # 广播到 [freq_len, embed_dim]
            self.complex_weight.data[:, :, 0] = decay.repeat(1, embed_dim) # 实部
            self.complex_weight.data[:, :, 1] = torch.randn(self.freq_len, embed_dim) * 0.01 # 虚部

    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
        Returns:
            Purified embeddings [batch_size, seq_len, embed_dim]
        """
        # 1. FFT 变换 (实数 -> 复数频谱)
        # x shape: [B, L, D] -> x_fft shape: [B, L//2+1, D]
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 2. 频谱滤波
        # 将参数视作复数: [freq_len, embed_dim]
        weight = torch.view_as_complex(self.complex_weight)
        
        # 广播机制应用滤波: [B, L//2+1, D] * [1, L//2+1, D] (unsqueeze for batch dim)
        x_filtered = x_fft * weight.unsqueeze(0)
        
        # 3. IFFT 逆变换 (复数频谱 -> 实数)
        x_restored = torch.fft.irfft(x_filtered, n=self.seq_len, dim=1, norm='ortho')
        
        # 4. 残差连接 (保留关键瞬态信息)
        return x + x_restored
