"""
PriorZero 稳定性优化模块 - 增强鲁棒版本

基于学者评审意见的关键改进：
1. ✅ 修复 Welford 无限记忆问题 - 使用 Batch Welford + EMA
2. ✅ 添加强制启动保底机制 - 防止死锁
3. ✅ 优化训练频率调节逻辑 - 避免恶性循环
4. ✅ 软性裁剪替代硬裁剪 - 保留稀疏奖励信号
5. ✅ 分布式训练安全保证 - 确保决策同步

核心改进：
- Adaptive Value Normalization - 自适应值归一化
- Curriculum Training Schedule - 课程式训练策略
- Gradient & Loss Monitoring - 梯度和损失监控
- Dynamic Training Frequency - 动态训练频率调整
- Distributed Training Safety - 分布式训练安全

Date: 2025-12-30
Version: 1.1 Enhanced
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import logging
import torch.distributed as dist

class AdaptiveValueNormalizer:
    """
    自适应值归一化器 - 增强版

    关键改进（基于学者评审）:
    1. ✅ Batch Welford + EMA 组合 - 避免无限记忆问题
    2. ✅ 软性裁剪（Log-Sym Transform）- 保留稀疏奖励信号
    3. ✅ 自适应动量 - 前期快速适应，后期稳定

    核心设计：
    - Welford 算法仅用于当前 batch 的精确计算（数值稳定）
    - EMA 用于全局统计量的更新（适应非平稳分布）
    - Log-Sym 变换而非硬裁剪（保留大奖励梯度）
    """

    def __init__(
        self,
        init_momentum: float = 0.9,          # 初始 EMA 动量
        final_momentum: float = 0.99,        # 最终 EMA 动量
        warmup_steps: int = 100,             # 热启动步数
        clip_method: str = 'soft',           # 'soft' (log-sym) or 'hard' (percentile)
        clip_percentile: float = 0.95,       # 硬裁剪百分位（仅当 clip_method='hard'）
        min_std: float = 1e-6,               # 最小标准差
        use_welford: bool = True,            # 使用 Welford 算法
    ):
        self.init_momentum = init_momentum
        self.final_momentum = final_momentum
        self.warmup_steps = warmup_steps
        self.clip_method = clip_method
        self.clip_percentile = clip_percentile
        self.min_std = min_std
        self.use_welford = use_welford

        # EMA 运行统计量（全局）
        self.running_mean = 0.0
        self.running_std = 1.0
        self.update_count = 0

        # History for monitoring
        self.value_history = deque(maxlen=1000)

        # Statistics
        self.stats = {
            'batch_means': [],
            'batch_stds': [],
            'running_means': [],
            'running_stds': [],
            'clipped_counts': [],
        }

    def get_current_momentum(self) -> float:
        """计算当前的 EMA 动量（从快到慢）"""
        if self.update_count >= self.warmup_steps:
            return self.final_momentum

        # Linear interpolation during warmup
        progress = self.update_count / self.warmup_steps
        return self.init_momentum + (self.final_momentum - self.init_momentum) * progress

    def compute_batch_stats_welford(self, values: np.ndarray) -> Tuple[float, float]:
        """
        使用 NumPy 的高精度实现替代 Python 循环
        NumPy 的 var 实现内部已经使用了数值稳定的算法
        """
        if len(values) == 0:
            return 0.0, 1.0
        
        # 使用 float64 进行计算以保证精度，避免 Python 循环
        values_f64 = values.astype(np.float64).flatten()
        mean = np.mean(values_f64)
        std = np.std(values_f64, ddof=1) # ddof=1 for unbiased estimator (n-1)
        
        return float(mean), max(float(std), self.min_std)
        
    # def compute_batch_stats_welford(self, values: np.ndarray) -> Tuple[float, float]:
    #     """
    #     使用 Welford 算法计算当前 batch 的精确统计量

    #     关键改进：仅用于 batch 内计算，不累积历史
    #     这样既保证数值稳定，又避免了"无限记忆"问题
    #     """
    #     if len(values) == 0:
    #         return 0.0, 1.0

    #     values_flat = values.flatten()
    #     n = len(values_flat)

    #     if n == 1:
    #         return float(values_flat[0]), self.min_std

    #     # Welford's algorithm for batch statistics
    #     mean = 0.0
    #     M2 = 0.0

    #     for i, value in enumerate(values_flat, 1):
    #         delta = value - mean
    #         mean += delta / i
    #         delta2 = value - mean
    #         M2 += delta * delta2

    #     variance = M2 / (n - 1) if n > 1 else 0.0
    #     std = np.sqrt(variance)

    #     return float(mean), max(float(std), self.min_std)

    def soft_clip_log_sym(self, values: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        软性裁剪 - Log-Symmetric Transform

        f(x) = sign(x) * log(1 + |x|)

        优势：
        - 保留大奖励的梯度信号（稀疏奖励友好）
        - 限制数值范围，防止梯度爆炸
        - 可微分，适合反向传播
        """
        original_values = values.copy()
        values_transformed = np.sign(values) * np.log1p(np.abs(values))

        # Count how many values were significantly transformed (|x| > 10)
        significant_transform = np.sum(np.abs(original_values) > 10)

        return values_transformed, significant_transform

    def hard_clip_percentile(self, values: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        硬裁剪 - 基于百分位数

        仅用于非稀疏奖励环境（如 Atari）
        """
        if self.update_count < 10:  # Skip clipping in early phase
            return values, 0

        percentile_low = (1 - self.clip_percentile) / 2 * 100
        percentile_high = (1 + self.clip_percentile) / 2 * 100
        lower_bound = np.percentile(values, percentile_low)
        upper_bound = np.percentile(values, percentile_high)

        original_values = values.copy()
        values_clipped = np.clip(values, lower_bound, upper_bound)
        clipped_count = np.sum(original_values != values_clipped)

        return values_clipped, clipped_count

    def normalize(
        self,
        values: torch.Tensor,
        clip_values: bool = True,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        归一化值 - 增强版

        流程：
        1. Welford 计算 batch 精确统计量
        2. EMA 更新全局运行统计量
        3. 软性/硬性裁剪
        4. 归一化

        Args:
            values: 输入值 [B] 或 [B, T]
            clip_values: 是否裁剪异常值
            return_stats: 是否返回统计信息
        """
        values_np = values.detach().cpu().numpy()

        # Step 1: Compute batch statistics with Welford (数值稳定)
        if self.use_welford:
            batch_mean, batch_std = self.compute_batch_stats_welford(values_np)
        else:
            batch_mean = float(values_np.mean())
            batch_std = max(float(values_np.std()), self.min_std)

        # Step 2: Apply clipping (optional)
        clipped_count = 0
        if clip_values:
            if self.clip_method == 'soft':
                values_np, clipped_count = self.soft_clip_log_sym(values_np)
                # Recompute stats after soft transform
                if self.use_welford:
                    batch_mean, batch_std = self.compute_batch_stats_welford(values_np)
                else:
                    batch_mean = float(values_np.mean())
                    batch_std = max(float(values_np.std()), self.min_std)
            elif self.clip_method == 'hard':
                values_np, clipped_count = self.hard_clip_percentile(values_np)
                # Recompute stats after hard clip
                if self.use_welford:
                    batch_mean, batch_std = self.compute_batch_stats_welford(values_np)
                else:
                    batch_mean = float(values_np.mean())
                    batch_std = max(float(values_np.std()), self.min_std)

        # Step 3: Update running statistics with EMA (适应非平稳)
        momentum = self.get_current_momentum()

        if self.update_count == 0:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
            self.running_std = momentum * self.running_std + (1 - momentum) * batch_std

        self.update_count += 1
        self.value_history.extend(values_np.flatten().tolist())

        # Step 4: Normalize
        normalized = (torch.from_numpy(values_np).to(values.device) - self.running_mean) / (self.running_std + self.min_std)

        # Record statistics
        stats = {
            'batch_mean': batch_mean,
            'batch_std': batch_std,
            'running_mean': self.running_mean,
            'running_std': self.running_std,
            'clipped_count': clipped_count,
            'total_count': len(values_np.flatten()),
            'current_momentum': momentum,
            'clip_method': self.clip_method,
        }

        # Store for analysis
        self.stats['batch_means'].append(batch_mean)
        self.stats['batch_stds'].append(batch_std)
        self.stats['running_means'].append(self.running_mean)
        self.stats['running_stds'].append(self.running_std)
        self.stats['clipped_counts'].append(clipped_count)

        if return_stats:
            return normalized, stats
        return normalized

    def get_summary_stats(self) -> Dict:
        """获取汇总统计信息"""
        if self.update_count == 0:
            return {}

        recent_n = min(100, len(self.value_history))
        recent_values = list(self.value_history)[-recent_n:]

        return {
            'total_updates': self.update_count,
            'current_mean': self.running_mean,
            'current_std': self.running_std,
            'recent_mean': np.mean(recent_values) if recent_values else 0.0,
            'recent_std': np.std(recent_values) if recent_values else 1.0,
            'recent_min': np.min(recent_values) if recent_values else 0.0,
            'recent_max': np.max(recent_values) if recent_values else 0.0,
            'clip_method': self.clip_method,
        }


class CurriculumTrainingScheduler:
    """
    课程式训练调度器 - 增强版

    关键改进（基于学者评审）:
    1. ✅ 强制启动保底机制 - 防止死锁
    2. ✅ 优化训练频率逻辑 - 避免恶性循环
    3. ✅ 阈值震荡防护 - 添加 hysteresis

    核心功能：
    1. World Model Warm-up: 先训练 WM 使其 value 预测准确
    2. Dynamic Training Ratio: 动态调整 WM/LLM 训练频率
    3. Progressive Difficulty: 逐步增加 LLM 训练频率
    4. Stability Monitoring: 监控训练稳定性自动调整
    """

    def __init__(
        self,
        wm_warmup_steps: int = 500,                 # WM 预热步数
        wm_warmup_update_ratio: int = 5,            # 预热期 WM 每次 collect 训练次数
        llm_start_threshold: float = 0.3,           # LLM 开始训练的 value 准确度阈值
        llm_start_hysteresis: float = 0.05,         # 阈值滞后（防止震荡）
        max_warmup_steps_hard_limit: int = 2000,    # 强制启动上限（防止死锁）
        min_llm_update_interval: int = 1,           # LLM 最小更新间隔
        max_llm_update_interval: int = 10,          # LLM 最大更新间隔
        min_llm_update_interval_hard: int = 1,      # LLM 最小更新间隔（硬限制，避免数据过时）
        stability_window: int = 50,                 # 稳定性监控窗口
    ):
        self.wm_warmup_steps = wm_warmup_steps
        self.wm_warmup_update_ratio = wm_warmup_update_ratio
        self.llm_start_threshold = llm_start_threshold
        self.llm_start_hysteresis = llm_start_hysteresis
        self.max_warmup_steps_hard_limit = max_warmup_steps_hard_limit
        self.min_llm_update_interval = min_llm_update_interval
        self.max_llm_update_interval = max_llm_update_interval
        self.min_llm_update_interval_hard = min_llm_update_interval_hard
        self.stability_window = stability_window

        # State
        self.current_step = 0
        self.llm_update_count = 0
        self.wm_update_count = 0

        # Metrics history
        self.value_prediction_errors = deque(maxlen=stability_window)
        self.llm_losses = deque(maxlen=stability_window)
        self.wm_losses = deque(maxlen=stability_window)

        # Flags
        self.llm_training_enabled = False
        self.llm_ever_enabled = False  # 防止震荡：一旦启用就不再禁用
        self.is_warming_up = True

    def should_train_wm(self, new_data_collected: bool = True) -> Tuple[bool, int]:
        """
        判断是否应该训练 World Model

        Returns:
            should_train: 是否训练
            num_updates: 训练次数
        """
        if not new_data_collected:
            return False, 0

        if self.is_warming_up:
            # Warm-up 期间每次都训练，且训练次数更多
            return True, self.wm_warmup_update_ratio
        else:
            # 正常训练期间
            return True, 1  # 或者根据配置返回 update_per_collect

    def should_train_llm(
        self,
        new_samples: int,
        value_prediction_accuracy: Optional[float] = None,
    ) -> Tuple[bool, Dict]:
        """
        判断是否应该训练 LLM - 增强版

        改进：
        1. ✅ 强制启动保底机制
        2. ✅ Hysteresis 防止震荡
        3. ✅ 最小频率限制避免数据过时

        Args:
            new_samples: 新收集的样本数
            value_prediction_accuracy: WM 的 value 预测准确度 (R^2 or 1 - MAPE)

        Returns:
            should_train: 是否训练
            info: 决策信息
        """
        info = {
            'reason': '',
            'current_step': self.current_step,
            'llm_training_enabled': self.llm_training_enabled,
            'is_warming_up': self.is_warming_up,
        }

        # ========================================================================
        # 强制启动保底机制（防止死锁）
        # ========================================================================
        if self.current_step >= self.max_warmup_steps_hard_limit and not self.llm_training_enabled:
            self.llm_training_enabled = True
            self.llm_ever_enabled = True
            self.is_warming_up = False
            logging.warning(
                f"⚠ Force-enabled LLM training at step {self.current_step} "
                f"(hard limit {self.max_warmup_steps_hard_limit} reached)"
            )
            info['reason'] = 'Force start: Max warmup steps reached'
            # 继续执行后续逻辑

        # ========================================================================
        # Check warm-up phase
        # ========================================================================
        if self.current_step < self.wm_warmup_steps:
            info['reason'] = f'WM warming up ({self.current_step}/{self.wm_warmup_steps})'
            return False, info

        # ========================================================================
        # Enable LLM training after warm-up (with hysteresis)
        # ========================================================================
        if not self.llm_training_enabled:
            if value_prediction_accuracy is not None:
                # 使用 hysteresis：启用时要求 >= threshold，禁用时要求 < threshold - hysteresis
                if value_prediction_accuracy >= self.llm_start_threshold:
                    self.llm_training_enabled = True
                    self.llm_ever_enabled = True
                    self.is_warming_up = False
                    logging.info(
                        f"✓ LLM training enabled! Value prediction accuracy: {value_prediction_accuracy:.3f} "
                        f">= threshold {self.llm_start_threshold}"
                    )
                else:
                    info['reason'] = (
                        f'Value accuracy {value_prediction_accuracy:.3f} < '
                        f'threshold {self.llm_start_threshold}'
                    )
                    return False, info
            else:
                # Fallback: enable after warmup even without accuracy metric
                self.llm_training_enabled = True
                self.llm_ever_enabled = True
                self.is_warming_up = False
                logging.warning("Value accuracy not provided, enabling LLM training after warmup")

        # ========================================================================
        # 防止震荡：一旦启用就不再禁用（除非明确重置）
        # ========================================================================
        if self.llm_ever_enabled and not self.llm_training_enabled:
            if value_prediction_accuracy is not None:
                # 只有在准确度严重下降时才禁用（使用 hysteresis）
                if value_prediction_accuracy >= self.llm_start_threshold - self.llm_start_hysteresis:
                    self.llm_training_enabled = True

        # Check sample count
        if new_samples <= 0:
            info['reason'] = 'No new samples'
            return False, info

        # Dynamic update interval based on stability
        update_interval = self._compute_llm_update_interval()

        # ========================================================================
        # 最小频率限制（防止数据过时）
        # ========================================================================
        update_interval = max(update_interval, self.min_llm_update_interval_hard)

        steps_since_last_update = self.current_step - self.llm_update_count * update_interval
        if steps_since_last_update >= update_interval:
            info['reason'] = f'Update interval reached ({steps_since_last_update} >= {update_interval})'
            return True, info
        else:
            info['reason'] = f'Waiting for update interval ({steps_since_last_update}/{update_interval})'
            return False, info

    def _compute_llm_update_interval(self) -> int:
        """
        根据训练稳定性动态计算 LLM 更新间隔 - 优化版

        改进：不应该单纯降低频率，而是通过其他方式稳定训练
        这里仅作为参考，实际稳定性应通过调整学习率、梯度裁剪等实现
        """
        if len(self.llm_losses) < self.stability_window // 2:
            # 初期使用默认间隔
            return self.min_llm_update_interval  # 改为最小间隔，确保充分训练

        # 计算损失的变化率（标准差）
        recent_losses = list(self.llm_losses)
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        if loss_mean == 0:
            cv = float('inf')  # Coefficient of variation
        else:
            cv = loss_std / (abs(loss_mean) + 1e-8)

        # CV 越小 -> 越稳定 -> 保持当前频率
        # CV 越大 -> 不稳定 -> 略微降低频率但不过分（防止数据过时）
        if cv < 0.1:  # Very stable
            interval = self.min_llm_update_interval
        elif cv < 0.3:  # Moderately stable
            interval = (self.min_llm_update_interval + self.max_llm_update_interval) // 2
        else:  # Unstable
            # 不过分降低频率，最多到 max 的一半
            interval = min(self.max_llm_update_interval // 2, self.max_llm_update_interval)

        return interval

    def record_metrics(
        self,
        wm_loss: Optional[float] = None,
        llm_loss: Optional[float] = None,
        value_pred_error: Optional[float] = None,
    ):
        """记录训练指标"""
        if wm_loss is not None:
            self.wm_losses.append(wm_loss)
            self.wm_update_count += 1

        if llm_loss is not None:
            self.llm_losses.append(llm_loss)
            self.llm_update_count += 1

        if value_pred_error is not None:
            self.value_prediction_errors.append(value_pred_error)

    def step(self):
        """前进一步"""
        self.current_step += 1

    def get_value_prediction_accuracy(self) -> Optional[float]:
        """
        计算 value 预测准确度

        Returns:
            accuracy: 准确度 (1 - MAPE)，范围 [0, 1]
        """
        if len(self.value_prediction_errors) == 0:
            return None

        # 使用 MAPE (Mean Absolute Percentage Error)
        mape = np.mean(list(self.value_prediction_errors))
        accuracy = max(0.0, 1.0 - mape)
        return accuracy

    def get_summary_stats(self) -> Dict:
        """获取汇总统计"""
        return {
            'current_step': self.current_step,
            'wm_update_count': self.wm_update_count,
            'llm_update_count': self.llm_update_count,
            'llm_training_enabled': self.llm_training_enabled,
            'llm_ever_enabled': self.llm_ever_enabled,
            'is_warming_up': self.is_warming_up,
            'value_prediction_accuracy': self.get_value_prediction_accuracy(),
            'recent_wm_loss': np.mean(list(self.wm_losses)) if self.wm_losses else None,
            'recent_llm_loss': np.mean(list(self.llm_losses)) if self.llm_losses else None,
            'llm_loss_stability': 1.0 / (np.std(list(self.llm_losses)) + 1e-8) if len(self.llm_losses) > 1 else 0.0,
        }


class StabilityMonitor:
    """
    训练稳定性监控器

    功能：
    1. 梯度监控：检测梯度爆炸/消失
    2. 损失监控：检测损失发散
    3. Value 分布监控：检测分布偏移
    4. 自动预警和建议
    """

    def __init__(
        self,
        grad_norm_threshold: float = 100.0,
        loss_spike_threshold: float = 3.0,
        value_shift_threshold: float = 2.0,
        window_size: int = 50,
    ):
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.value_shift_threshold = value_shift_threshold
        self.window_size = window_size

        # History
        self.wm_grad_norms = deque(maxlen=window_size)
        self.llm_grad_norms = deque(maxlen=window_size)
        self.wm_losses = deque(maxlen=window_size)
        self.llm_losses = deque(maxlen=window_size)
        self.value_means = deque(maxlen=window_size)
        self.value_stds = deque(maxlen=window_size)

        # Warnings
        self.warnings = []

    def check_gradient_health(
        self,
        wm_grad_norm: Optional[float] = None,
        llm_grad_norm: Optional[float] = None,
    ) -> Dict[str, any]:
        """检查梯度健康状态"""
        issues = []

        if wm_grad_norm is not None:
            self.wm_grad_norms.append(wm_grad_norm)

            if wm_grad_norm > self.grad_norm_threshold:
                issues.append(f"WM gradient exploding: {wm_grad_norm:.2f}")
            elif wm_grad_norm < 1e-6:
                issues.append(f"WM gradient vanishing: {wm_grad_norm:.2e}")

        if llm_grad_norm is not None:
            self.llm_grad_norms.append(llm_grad_norm)

            if llm_grad_norm > self.grad_norm_threshold:
                issues.append(f"LLM gradient exploding: {llm_grad_norm:.2f}")
            elif llm_grad_norm < 1e-6:
                issues.append(f"LLM gradient vanishing: {llm_grad_norm:.2e}")

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'wm_grad_norm': wm_grad_norm,
            'llm_grad_norm': llm_grad_norm,
        }

    def check_loss_stability(
        self,
        wm_loss: Optional[float] = None,
        llm_loss: Optional[float] = None,
    ) -> Dict[str, any]:
        """检查损失稳定性"""
        issues = []

        if wm_loss is not None:
            self.wm_losses.append(wm_loss)

            if len(self.wm_losses) > 10:
                recent_mean = np.mean(list(self.wm_losses)[:-1])
                if wm_loss > recent_mean * self.loss_spike_threshold:
                    issues.append(f"WM loss spike: {wm_loss:.4f} vs mean {recent_mean:.4f}")

        if llm_loss is not None:
            self.llm_losses.append(llm_loss)

            if len(self.llm_losses) > 10:
                recent_mean = np.mean(list(self.llm_losses)[:-1])
                if llm_loss > recent_mean * self.loss_spike_threshold:
                    issues.append(f"LLM loss spike: {llm_loss:.4f} vs mean {recent_mean:.4f}")

        return {
            'stable': len(issues) == 0,
            'issues': issues,
            'wm_loss': wm_loss,
            'llm_loss': llm_loss,
        }

    def check_value_distribution(
        self,
        value_mean: float,
        value_std: float,
    ) -> Dict[str, any]:
        """检查 value 分布是否异常"""
        issues = []

        self.value_means.append(value_mean)
        self.value_stds.append(value_std)

        if len(self.value_means) > 10:
            historical_mean = np.mean(list(self.value_means)[:-1])
            historical_std = np.mean(list(self.value_stds)[:-1])

            mean_shift = abs(value_mean - historical_mean) / (abs(historical_mean) + 1e-8)
            std_shift = abs(value_std - historical_std) / (historical_std + 1e-8)

            if mean_shift > self.value_shift_threshold:
                issues.append(f"Value mean shift: {value_mean:.2f} vs historical {historical_mean:.2f}")

            if std_shift > self.value_shift_threshold:
                issues.append(f"Value std shift: {value_std:.2f} vs historical {historical_std:.2f}")

        return {
            'normal': len(issues) == 0,
            'issues': issues,
            'value_mean': value_mean,
            'value_std': value_std,
        }

    def get_recommendations(self) -> list[str]:
        """基于监控结果给出优化建议"""
        recommendations = []

        # Check gradient norms
        if len(self.wm_grad_norms) > 10:
            recent_wm_grad = np.mean(list(self.wm_grad_norms)[-10:])
            if recent_wm_grad > self.grad_norm_threshold * 0.5:
                recommendations.append(
                    f"WM gradient norm high ({recent_wm_grad:.2f}). "
                    f"Consider: (1) Reduce learning rate, (2) Increase gradient clipping"
                )

        if len(self.llm_grad_norms) > 10:
            recent_llm_grad = np.mean(list(self.llm_grad_norms)[-10:])
            if recent_llm_grad > self.grad_norm_threshold * 0.5:
                recommendations.append(
                    f"LLM gradient norm high ({recent_llm_grad:.2f}). "
                    f"Consider: (1) Reduce LLM learning rate, (2) Use gradient accumulation"
                )

        # Check loss trends
        if len(self.wm_losses) > 20:
            early_mean = np.mean(list(self.wm_losses)[:10])
            recent_mean = np.mean(list(self.wm_losses)[-10:])
            if recent_mean > early_mean * 1.5:
                recommendations.append(
                    f"WM loss increasing trend. "
                    f"Consider: (1) Check data quality, (2) Reduce learning rate, (3) Add regularization"
                )

        if len(self.llm_losses) > 20:
            early_mean = np.mean(list(self.llm_losses)[:10])
            recent_mean = np.mean(list(self.llm_losses)[-10:])
            if recent_mean > early_mean * 1.5:
                recommendations.append(
                    f"LLM loss increasing trend. "
                    f"Consider: (1) Pause LLM training temporarily, (2) Improve value normalization"
                )

        return recommendations

    def get_summary(self) -> Dict:
        """获取监控摘要"""
        summary = {
            'total_checks': len(self.wm_grad_norms),
            'warnings': self.warnings[-10:] if self.warnings else [],
            'recommendations': self.get_recommendations(),
        }

        if len(self.wm_grad_norms) > 0:
            summary['wm_grad_norm_mean'] = np.mean(list(self.wm_grad_norms))
            summary['wm_grad_norm_max'] = np.max(list(self.wm_grad_norms))

        if len(self.llm_grad_norms) > 0:
            summary['llm_grad_norm_mean'] = np.mean(list(self.llm_grad_norms))
            summary['llm_grad_norm_max'] = np.max(list(self.llm_grad_norms))

        if len(self.wm_losses) > 0:
            summary['wm_loss_mean'] = np.mean(list(self.wm_losses))
            summary['wm_loss_std'] = np.std(list(self.wm_losses))

        if len(self.llm_losses) > 0:
            summary['llm_loss_mean'] = np.mean(list(self.llm_losses))
            summary['llm_loss_std'] = np.std(list(self.llm_losses))

        return summary


class PriorZeroStabilityOptimizer:
    """
    PriorZero 综合稳定性优化器 - 增强版

    整合所有优化模块，提供统一接口

    增强特性：
    - 支持 Batch Welford + EMA 组合
    - 支持软性/硬性裁剪模式
    - 支持强制启动保底机制
    - 支持阈值滞后防震荡
    - 完整的训练稳定性监控
    """

    def __init__(
        self,
        # Value normalization config - Enhanced
        value_norm_init_momentum: float = 0.9,
        value_norm_final_momentum: float = 0.99,
        value_norm_warmup_steps: int = 100,
        value_norm_clip_method: str = 'soft',           # NEW: 'soft' or 'hard'
        value_norm_clip_percentile: float = 0.95,
        value_norm_use_welford: bool = True,

        # Curriculum training config - Enhanced
        wm_warmup_steps: int = 500,
        wm_warmup_update_ratio: int = 5,
        llm_start_threshold: float = 0.3,
        llm_start_hysteresis: float = 0.05,             # NEW: Hysteresis
        max_warmup_steps_hard_limit: int = 2000,        # NEW: Force-start limit
        min_llm_update_interval: int = 1,
        max_llm_update_interval: int = 10,
        min_llm_update_interval_hard: int = 1,          # NEW: Hard limit

        # Stability monitoring config
        grad_norm_threshold: float = 100.0,
        loss_spike_threshold: float = 3.0,
        value_shift_threshold: float = 2.0,

        # Logging
        log_interval: int = 10,
        rank: int = 0,
    ):
        self.rank = rank
        self.log_interval = log_interval
        self.step_count = 0

        # Initialize modules with enhanced parameters
        self.value_normalizer = AdaptiveValueNormalizer(
            init_momentum=value_norm_init_momentum,
            final_momentum=value_norm_final_momentum,
            warmup_steps=value_norm_warmup_steps,
            clip_method=value_norm_clip_method,
            clip_percentile=value_norm_clip_percentile,
            use_welford=value_norm_use_welford,
        )

        self.curriculum_scheduler = CurriculumTrainingScheduler(
            wm_warmup_steps=wm_warmup_steps,
            wm_warmup_update_ratio=wm_warmup_update_ratio,
            llm_start_threshold=llm_start_threshold,
            llm_start_hysteresis=llm_start_hysteresis,
            max_warmup_steps_hard_limit=max_warmup_steps_hard_limit,
            min_llm_update_interval=min_llm_update_interval,
            max_llm_update_interval=max_llm_update_interval,
            min_llm_update_interval_hard=min_llm_update_interval_hard,
        )

        self.stability_monitor = StabilityMonitor(
            grad_norm_threshold=grad_norm_threshold,
            loss_spike_threshold=loss_spike_threshold,
            value_shift_threshold=value_shift_threshold,
        )

        logging.info(f"[Rank {rank}] PriorZero Stability Optimizer (Enhanced v1.1) initialized")
        logging.info(f"  - WM warmup steps: {wm_warmup_steps}")
        logging.info(f"  - LLM start threshold: {llm_start_threshold} (hysteresis: {llm_start_hysteresis})")
        logging.info(f"  - Max warmup hard limit: {max_warmup_steps_hard_limit}")
        logging.info(f"  - Value norm: {value_norm_clip_method} clipping, Welford={value_norm_use_welford}")

    def normalize_target_values(
        self,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """归一化 target values"""
        normalized, stats = self.value_normalizer.normalize(
            target_values,
            clip_values=True,
            return_stats=True,
        )

        # Monitor value distribution
        self.stability_monitor.check_value_distribution(
            value_mean=stats['batch_mean'],
            value_std=stats['batch_std'],
        )

        return normalized, stats

    def should_train_world_model(self, new_data_collected: bool = True) -> Tuple[bool, int]:
        """判断是否训练 World Model"""
        return self.curriculum_scheduler.should_train_wm(new_data_collected)

    def should_train_llm(
        self,
        new_samples: int,
        value_prediction_error: Optional[float] = None,
    ) -> Tuple[bool, Dict]:
        """判断是否训练 LLM"""
        # Compute value prediction accuracy
        accuracy = None
        if value_prediction_error is not None:
            self.curriculum_scheduler.record_metrics(value_pred_error=value_prediction_error)
            accuracy = self.curriculum_scheduler.get_value_prediction_accuracy()

        should_train, info = self.curriculum_scheduler.should_train_llm(
            new_samples=new_samples,
            value_prediction_accuracy=accuracy,
        )

        # DDP Sync TODO
        if dist.is_initialized():
            decision = torch.tensor([float(should_train)], device='cuda')
            dist.all_reduce(decision, op=dist.ReduceOp.MAX) # 只要有一个 Rank 想练，大家就一起练? 或者 MIN
            # 建议使用广播：以 Rank 0 为准
            # dist.broadcast(decision, src=0)
            should_train = bool(decision.item() > 0.5)
            # 注意：如果强制同步，info 中的 reason 可能会在不同 rank 不一致，仅供参考
    

        return should_train, info

    def record_training_metrics(
        self,
        wm_loss: Optional[float] = None,
        wm_grad_norm: Optional[float] = None,
        llm_loss: Optional[float] = None,
        llm_grad_norm: Optional[float] = None,
    ):
        """记录训练指标"""
        # Record to curriculum scheduler
        self.curriculum_scheduler.record_metrics(
            wm_loss=wm_loss,
            llm_loss=llm_loss,
        )

        # Monitor stability
        if wm_grad_norm is not None or llm_grad_norm is not None:
            grad_health = self.stability_monitor.check_gradient_health(
                wm_grad_norm=wm_grad_norm,
                llm_grad_norm=llm_grad_norm,
            )
            if not grad_health['healthy'] and self.rank == 0:
                logging.warning(f"⚠ Gradient health issues: {grad_health['issues']}")

        if wm_loss is not None or llm_loss is not None:
            loss_health = self.stability_monitor.check_loss_stability(
                wm_loss=wm_loss,
                llm_loss=llm_loss,
            )
            if not loss_health['stable'] and self.rank == 0:
                logging.warning(f"⚠ Loss stability issues: {loss_health['issues']}")

    def step(self):
        """前进一步"""
        self.step_count += 1
        self.curriculum_scheduler.step()

        # Periodic logging
        if self.rank == 0 and self.step_count % self.log_interval == 0:
            self._log_status()

    def _log_status(self):
        """记录当前状态"""
        curriculum_stats = self.curriculum_scheduler.get_summary_stats()
        value_stats = self.value_normalizer.get_summary_stats()
        stability_summary = self.stability_monitor.get_summary()

        logging.info("="*80)
        logging.info(f"PriorZero Stability Optimizer Status (Step {self.step_count})")
        logging.info("="*80)

        logging.info("Curriculum Training:")
        logging.info(f"  - WM updates: {curriculum_stats['wm_update_count']}")
        logging.info(f"  - LLM updates: {curriculum_stats['llm_update_count']}")
        logging.info(f"  - LLM training enabled: {curriculum_stats['llm_training_enabled']}")
        logging.info(f"  - LLM ever enabled: {curriculum_stats.get('llm_ever_enabled', False)}")
        logging.info(f"  - Is warming up: {curriculum_stats['is_warming_up']}")
        if curriculum_stats['value_prediction_accuracy'] is not None:
            logging.info(f"  - Value prediction accuracy: {curriculum_stats['value_prediction_accuracy']:.3f}")

        logging.info("\nValue Normalization:")
        logging.info(f"  - Total updates: {value_stats.get('total_updates', 0)}")
        logging.info(f"  - Clip method: {value_stats.get('clip_method', 'N/A')}")
        logging.info(f"  - Current mean: {value_stats.get('current_mean', 0.0):.3f}")
        logging.info(f"  - Current std: {value_stats.get('current_std', 1.0):.3f}")
        logging.info(f"  - Recent mean: {value_stats.get('recent_mean', 0.0):.3f}")
        logging.info(f"  - Recent range: [{value_stats.get('recent_min', 0.0):.2f}, {value_stats.get('recent_max', 0.0):.2f}]")

        logging.info("\nStability Monitoring:")
        if 'wm_grad_norm_mean' in stability_summary:
            logging.info(f"  - WM grad norm (mean/max): {stability_summary['wm_grad_norm_mean']:.2f} / {stability_summary['wm_grad_norm_max']:.2f}")
        if 'llm_grad_norm_mean' in stability_summary:
            logging.info(f"  - LLM grad norm (mean/max): {stability_summary['llm_grad_norm_mean']:.2f} / {stability_summary['llm_grad_norm_max']:.2f}")
        if 'wm_loss_mean' in stability_summary:
            logging.info(f"  - WM loss (mean±std): {stability_summary['wm_loss_mean']:.4f} ± {stability_summary['wm_loss_std']:.4f}")
        if 'llm_loss_mean' in stability_summary:
            logging.info(f"  - LLM loss (mean±std): {stability_summary['llm_loss_mean']:.4f} ± {stability_summary['llm_loss_std']:.4f}")

        if stability_summary['recommendations']:
            logging.info("\n🔔 Recommendations:")
            for i, rec in enumerate(stability_summary['recommendations'], 1):
                logging.info(f"  {i}. {rec}")

        logging.info("="*80 + "\n")

    def get_tb_metrics(self) -> Dict:
        """获取用于 TensorBoard 记录的指标"""
        curriculum_stats = self.curriculum_scheduler.get_summary_stats()
        value_stats = self.value_normalizer.get_summary_stats()

        metrics = {
            'stability/wm_update_count': curriculum_stats['wm_update_count'],
            'stability/llm_update_count': curriculum_stats['llm_update_count'],
            'stability/llm_training_enabled': int(curriculum_stats['llm_training_enabled']),
            'stability/llm_ever_enabled': int(curriculum_stats.get('llm_ever_enabled', False)),
            'stability/is_warming_up': int(curriculum_stats['is_warming_up']),

            'stability/value_norm_mean': value_stats.get('current_mean', 0.0),
            'stability/value_norm_std': value_stats.get('current_std', 1.0),
            'stability/value_recent_mean': value_stats.get('recent_mean', 0.0),
            'stability/value_recent_std': value_stats.get('recent_std', 1.0),
        }

        if curriculum_stats['value_prediction_accuracy'] is not None:
            metrics['stability/value_prediction_accuracy'] = curriculum_stats['value_prediction_accuracy']

        if curriculum_stats['recent_wm_loss'] is not None:
            metrics['stability/wm_loss_recent'] = curriculum_stats['recent_wm_loss']

        if curriculum_stats['recent_llm_loss'] is not None:
            metrics['stability/llm_loss_recent'] = curriculum_stats['recent_llm_loss']

        if curriculum_stats['llm_loss_stability'] is not None:
            metrics['stability/llm_loss_stability'] = curriculum_stats['llm_loss_stability']

        return metrics
