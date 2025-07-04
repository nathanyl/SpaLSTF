import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

from ray.air import session
import os

from .SpaLSTF_scheduler import NoiseScheduler

# class WeightedSampler:
#     def __init__(self, num_timesteps):
#         self.num_timesteps = num_timesteps
#         # 初始化每个时间步的权重为均匀分布
#         self.weights = torch.ones(num_timesteps)
#
#     def update_weights(self, timestep_losses):
#         # 使用EMA更新每个时间步的权重
#         self.weights = 0.98 * self.weights + 0.02 * timestep_losses
#
#     def sample(self, batch_size):
#         # 根据权重生成加权采样的时间步
#         probabilities = self.weights / self.weights.sum()
#         timesteps = torch.multinomial(probabilities, batch_size, replacement=True)
#         return timesteps
# class MultivariateKLDivergence:
#     def __init__(self, regularization=1e-6):
#         """
#         用于计算多元正态分布之间的 KL 散度的类。
#
#         参数:
#         - regularization: float, 协方差矩阵的正则化项，确保协方差矩阵可逆。
#         """
#         self.regularization = regularization
#
#     def compute(self, pred_matrix, true_matrix):
#         """
#         计算 pred_matrix 和 true_matrix 的多元正态分布 KL 散度。
#
#         参数:
#         - pred_matrix: torch.Tensor, 预测值矩阵，形状为 [n, g]
#         - true_matrix: torch.Tensor, 真实值矩阵，形状为 [n, g]
#
#         返回:
#         - kl_divergence: float, KL 散度值
#         """
#         # 计算均值和协方差矩阵
#         mean_pred = pred_matrix.mean(dim=0)
#         mean_true = true_matrix.mean(dim=0)
#
#         cov_pred = torch.cov(pred_matrix.T) + torch.eye(pred_matrix.shape[1]) * self.regularization
#         cov_true = torch.cov(true_matrix.T) + torch.eye(true_matrix.shape[1]) * self.regularization
#
#         # 计算 KL 散度的各项
#         inv_cov_true = torch.linalg.inv(cov_true)
#         trace_term = torch.trace(inv_cov_true @ cov_pred)
#
#         mean_diff = (mean_true - mean_pred).unsqueeze(1)  # 确保为列向量
#         mahalanobis_term = (mean_diff.T @ inv_cov_true @ mean_diff).item()
#
#         log_det_ratio = torch.logdet(cov_true) - torch.logdet(cov_pred)
#
#         # 计算最终 KL 散度
#         kl_divergence = 0.5 * (trace_term + mahalanobis_term - pred_matrix.shape[1] + log_det_ratio)
#
#         return kl_divergence
#

def normal_train_SpaLSTF(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask = None,
                 edge_index = None):
    """

    Args:
        lr (float): learning rate 
        pred_type (str, optional): noise or x_0. Defaults to 'noise'.
        diffusion_step (int, optional): timestep. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:1').
        is_tqdm (bool, optional): tqdm. Defaults to True.
        is_tune (bool, optional):  ray tune. Defaults to False.

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    mse_criterion = nn.MSELoss()
    # kl_criterion = MultivariateKLDivergence(regularization=1e-6)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()


    # sampler = WeightedSampler(diffusion_step)
    # for epoch in t_epoch:
    #     epoch_loss = 0.
    #     for i, (x, x_cond) in enumerate(dataloader):
    #         x, x_cond = x.float().to(device), x_cond.float().to(device)
    #
    #         # 添加噪声和采样时间步
    #         noise = torch.randn(x.shape).to(device)
    #         timesteps = sampler.sample(x.shape[0]).long()  # 使用加权采样器进行采样
    #
    #         x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)
    #         mask = torch.tensor(mask).to(device)
    #         x_noisy = x_t * (1 - mask) + x * mask
    #
    #         # 模型预测
    #         noise_preds = model(x_noisy, t=timesteps.to(device), y=x_cond)
    #
    #         # 批量计算每个样本的损失（使用无缩减的MSELoss）
    #         losses = criterion(noise * (1 - mask), noise_preds * (1 - mask))
    #         losses = losses.mean(dim=1)  # 对每个样本的特征维度求均值, 结果是形状 [batch_size]
    #
    #         # 初始化存储每个时间步总损失和样本数量的张量
    #         timestep_losses = torch.zeros(diffusion_step, device=device)
    #         timestep_counts = torch.zeros(diffusion_step, device=device)
    #
    #         # 将每个样本的损失按时间步汇总
    #         timestep_losses.scatter_add_(0, timesteps, losses)
    #         timestep_counts.scatter_add_(0, timesteps, torch.ones_like(losses))
    #
    #         # 计算每个时间步的平均损失，将未采样的时间步设置为均值
    #         sampled_losses = timestep_losses[timestep_counts > 0]
    #         mean_loss = sampled_losses.mean() if sampled_losses.numel() > 0 else 0.0
    #         timestep_losses[timestep_counts == 0] = mean_loss
    #
    #         # 更新加权采样器
    #         sampler.update_weights(timestep_losses.detach())
    #
    #         # 计算总体损失并进行反向传播
    #         overall_loss = losses.mean()
    #         overall_loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         epoch_loss += overall_loss.item()
    #
    #     # 输出每个epoch的损失
    #     epoch_loss = epoch_loss / (i + 1)  # 计算当前轮次的平均损失
    #     if is_tqdm:
    #         t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')
    #     if is_tune:
    #         session.report({'loss': epoch_loss})

    # for epoch in t_epoch:
    #     epoch_loss = 0.
    #     for i, (x, x_cond) in enumerate(dataloader):
    #         x, x_cond = x.float().to(device), x_cond.float().to(device)
    #
    #         noise = torch.randn(x.shape).to(device)
    #         timesteps = sampler.sample(x.shape[0]).long()  # 使用加权采样器进行采样
    #
    #         x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)
    #
    #         mask = torch.tensor(mask).to(device)
    #         x_noisy = x_t * (1 - mask) + x * mask
    #
    #         noise_preds = model(x_noisy, t=timesteps.to(device), y=x_cond)
    #
    #         # 初始化一个用于存储每个时间步损失的列表
    #         timestep_losses = torch.zeros(diffusion_step).to(device)
    #         timestep_counts = torch.zeros(diffusion_step).to(device)
    #
    #         # 计算每个采样到的时间步的损失
    #         for sample_index in range(x.shape[0]):
    #             t = timesteps[sample_index]  # 获取当前样本的时间步
    #             # 计算当前样本的损失
    #             loss_t = criterion(noise[sample_index] * (1 - mask), noise_preds[sample_index] * (1 - mask))
    #
    #             # 将损失累加到对应的时间步
    #             timestep_losses[t] += loss_t
    #             timestep_counts[t] += 1  # 记录当前时间步的样本数量
    #
    #         # 计算已采样时间步的平均损失
    #         sampled_losses = timestep_losses[timestep_counts > 0]
    #         if sampled_losses.numel() > 0:  # 如果有采样到的损失
    #             mean_loss = sampled_losses.mean()  # 计算已采样时间步的均值
    #         else:
    #             mean_loss = 0.0  # 如果没有采样到的损失，设为0（或其他合适值）
    #

    #
    #         # 更新加权采样器的权重
    #         sampler.update_weights(timestep_losses.detach())
    #
    #         # 计算总体损失
    #         overall_loss = criterion(noise * (1 - mask), noise_preds * (1 - mask))
    #
    #         overall_loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防止梯度爆炸
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         epoch_loss += overall_loss.item()
    #
    #     epoch_loss = epoch_loss / (i + 1)  # 计算当前轮次的平均损失
    #     if is_tqdm:
    #         t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # 更新进度条
    #     if is_tune:
    #         session.report({'loss': epoch_loss})  # 如果使用 Ray Tune，则报告损失

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond) in enumerate(dataloader):
            x, x_cond = x.float().to(device), x_cond.float().to(device)
            # celltype = celltype.to(device)

            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()

            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps)

            mask = torch.tensor(mask).to(device)
            x_noisy = x_t * (1 - mask) + x * mask

            noise_pred = model(x_noisy, t=timesteps.to(device), y=x_cond, edge_index=edge_index)
            # loss = criterion(noise_pred, noise)
            mse_loss = mse_criterion(noise*(1-mask), noise_pred*(1-mask))
            # kl_loss = kl_criterion(noise*(1-mask), noise_pred*(1-mask))
            kl_loss = torch.distributions.kl_divergence(
                torch.distributions.Normal(noise_pred.mean(), noise_pred.std() + 1e-6),  # 预测分布
                torch.distributions.Normal(noise.mean(), noise.std() + 1e-6)  # 目标分布
            )
            if x.shape[1] < 120:
                alpha = 2.8
            else:
                alpha = 0.02
            # 将 MSE 损失和 KL 散度损失合并，权重可以根据实验调整
            loss = mse_loss + alpha * kl_loss  # alpha 为 KL 损失的权重因子

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})




