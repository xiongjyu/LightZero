"""
PriorZero 稳定性优化模块

核心优化：
1. Adaptive Value Normalization - 自适应值归一化
2. Curriculum Training Schedule - 课程式训练策略
3. Gradient & Loss Monitoring - 梯度和损失监控
4. Dynamic Training Frequency - 动态训练频率调整

Date: 2025-12-30
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import logging

class AdaptiveValueNormalizer:
    """
    自适应值归一化器

    核心改进：
    1. 使用 Welford's online algorithm 保证数值稳定性
    2. 前期使用更快的适应速度，后期使用更慢的稳定速度
    3. 自动检测异常值并调整
    4. 支持分位数裁剪防止极端值影响
    """

    def __init__(
        self,
        init_momentum: float = 0.9,      # 初始 EMA 动量
        final_momentum: float = 0.99,    # 最终 EMA 动量
        warmup_steps: int = 100,         # 热启动步数
        clip_percentile: float = 0.95,   # 异常值裁剪百分位
        min_std: float = 1e-6,           # 最小标准差
        use_welford: bool = True,        # 使用 Welford 算法
    ):
        self.init_momentum = init_momentum
        self.final_momentum = final_momentum
        self.warmup_steps = warmup_steps
        self.clip_percentile = clip_percentile
        self.min_std = min_std
        self.use_welford = use_welford

        # Welford's online algorithm state
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean

        # EMA state (backup)
        self.ema_mean = 0.0
        self.ema_std = 1.0

        # History for anomaly detection
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
        if self.count >= self.warmup_steps:
            return self.final_momentum

        # Linear interpolation during warmup
        progress = self.count / self.warmup_steps
        return self.init_momentum + (self.final_momentum - self.init_momentum) * progress

    def update_welford(self, values: np.ndarray) -> Tuple[float, float]:
        """
        使用 Welford's online algorithm 更新统计量
        数值稳定性更好，适合处理大规模数据
        """
        for value in values.flatten():
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.M2 += delta * delta2

        if self.count < 2:
            return self.mean, 1.0

        variance = self.M2 / (self.count - 1)
        std = np.sqrt(variance)
        return self.mean, max(std, self.min_std)

    def update_ema(self, batch_mean: float, batch_std: float) -> Tuple[float, float]:
        """使用 EMA 更新统计量（原方法）"""
        momentum = self.get_current_momentum()

        if self.count == 0:
            self.ema_mean = batch_mean
            self.ema_std = max(batch_std, self.min_std)
        else:
            self.ema_mean = momentum * self.ema_mean + (1 - momentum) * batch_mean
            self.ema_std = momentum * self.ema_std + (1 - momentum) * max(batch_std, self.min_std)

        return self.ema_mean, self.ema_std

    def normalize(
        self,
        values: torch.Tensor,
        clip_values: bool = True,
        return_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        归一化值

        Args:
            values: 输入值 [B] 或 [B, T]
            clip_values: 是否裁剪异常值
            return_stats: 是否返回统计信息

        Returns:
            normalized_values: 归一化后的值
            stats (optional): 统计信息字典
        """
        values_np = values.detach().cpu().numpy()
        batch_mean = values_np.mean()
        batch_std = values_np.std()

        # Clip outliers before normalization
        clipped_count = 0
        if clip_values and self.count > 10:  # Only clip after some warmup
            percentile_low = (1 - self.clip_percentile) / 2 * 100
            percentile_high = (1 + self.clip_percentile) / 2 * 100
            lower_bound = np.percentile(values_np, percentile_low)
            upper_bound = np.percentile(values_np, percentile_high)

            original_values = values_np.copy()
            values_np = np.clip(values_np, lower_bound, upper_bound)
            clipped_count = np.sum(original_values != values_np)

            # Recompute batch stats after clipping
            batch_mean = values_np.mean()
            batch_std = values_np.std()

        # Update statistics
        if self.use_welford:
            running_mean, running_std = self.update_welford(values_np)
        else:
            running_mean, running_std = self.update_ema(batch_mean, batch_std)

        self.count += 1
        self.value_history.extend(values_np.flatten().tolist())

        # Normalize
        normalized = (values - running_mean) / (running_std + self.min_std)

        # Record statistics
        stats = {
            'batch_mean': batch_mean,
            'batch_std': batch_std,
            'running_mean': running_mean,
            'running_std': running_std,
            'clipped_count': clipped_count,
            'total_count': len(values_np.flatten()),
            'current_momentum': self.get_current_momentum(),
        }

        # Store for analysis
        self.stats['batch_means'].append(batch_mean)
        self.stats['batch_stds'].append(batch_std)
        self.stats['running_means'].append(running_mean)
        self.stats['running_stds'].append(running_std)
        self.stats['clipped_counts'].append(clipped_count)

        if return_stats:
            return normalized, stats
        return normalized

    def get_summary_stats(self) -> Dict:
        """获取汇总统计信息"""
        if self.count == 0:
            return {}

        recent_n = min(100, len(self.value_history))
        recent_values = list(self.value_history)[-recent_n:]

        return {
            'total_updates': self.count,
            'current_mean': self.mean if self.use_welford else self.ema_mean,
            'current_std': np.sqrt(self.M2 / (self.count - 1)) if self.use_welford and self.count > 1 else self.ema_std,
            'recent_mean': np.mean(recent_values) if recent_values else 0.0,
            'recent_std': np.std(recent_values) if recent_values else 1.0,
            'recent_min': np.min(recent_values) if recent_values else 0.0,
            'recent_max': np.max(recent_values) if recent_values else 0.0,
        }


class CurriculumTrainingScheduler:
    """
    课程式训练调度器

    核心功能：
    1. World Model Warm-up: 先训练 WM 使其 value 预测准确
    2. Dynamic Training Ratio: 动态调整 WM/LLM 训练频率
    3. Progressive Difficulty: 逐步增加 LLM 训练难度
    4. Stability Monitoring: 监控训练稳定性自动调整
    """

    def __init__(
        self,
        wm_warmup_steps: int = 500,           # WM 预热步数
        wm_warmup_update_ratio: int = 5,      # 预热期 WM 每次 collect 训练次数
        llm_start_threshold: float = 0.3,     # LLM 开始训练的 value 准确度阈值
        min_llm_update_interval: int = 1,     # LLM 最小更新间隔
        max_llm_update_interval: int = 10,    # LLM 最大更新间隔
        stability_window: int = 50,           # 稳定性监控窗口
    ):
        self.wm_warmup_steps = wm_warmup_steps
        self.wm_warmup_update_ratio = wm_warmup_update_ratio
        self.llm_start_threshold = llm_start_threshold
        self.min_llm_update_interval = min_llm_update_interval
        self.max_llm_update_interval = max_llm_update_interval
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
        判断是否应该训练 LLM

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

        # Check warm-up phase
        if self.current_step < self.wm_warmup_steps:
            info['reason'] = f'WM warming up ({self.current_step}/{self.wm_warmup_steps})'
            return False, info

        # Enable LLM training after warm-up
        if not self.llm_training_enabled:
            if value_prediction_accuracy is not None:
                if value_prediction_accuracy >= self.llm_start_threshold:
                    self.llm_training_enabled = True
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
                self.is_warming_up = False
                logging.warning("Value accuracy not provided, enabling LLM training after warmup")

        # Check sample count
        if new_samples <= 0:
            info['reason'] = 'No new samples'
            return False, info

        # Dynamic update interval based on stability
        update_interval = self._compute_llm_update_interval()

        steps_since_last_update = self.current_step - self.llm_update_count * update_interval
        if steps_since_last_update >= update_interval:
            info['reason'] = f'Update interval reached ({steps_since_last_update} >= {update_interval})'
            return True, info
        else:
            info['reason'] = f'Waiting for update interval ({steps_since_last_update}/{update_interval})'
            return False, info

    def _compute_llm_update_interval(self) -> int:
        """
        根据训练稳定性动态计算 LLM 更新间隔

        稳定 -> 更频繁更新
        不稳定 -> 降低更新频率
        """
        if len(self.llm_losses) < self.stability_window // 2:
            # 初期使用默认间隔
            return self.max_llm_update_interval

        # 计算损失的变化率（标准差）
        recent_losses = list(self.llm_losses)
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        if loss_mean == 0:
            cv = float('inf')  # Coefficient of variation
        else:
            cv = loss_std / (abs(loss_mean) + 1e-8)

        # CV 越小 -> 越稳定 -> 更新间隔越小
        # CV 越大 -> 不稳定 -> 更新间隔越大
        if cv < 0.1:  # Very stable
            interval = self.min_llm_update_interval
        elif cv < 0.3:  # Moderately stable
            interval = (self.min_llm_update_interval + self.max_llm_update_interval) // 2
        else:  # Unstable
            interval = self.max_llm_update_interval

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
    PriorZero 综合稳定性优化器

    整合所有优化模块，提供统一接口
    """

    def __init__(
        self,
        # Value normalization config
        value_norm_init_momentum: float = 0.9,
        value_norm_final_momentum: float = 0.99,
        value_norm_warmup_steps: int = 100,
        value_norm_clip_percentile: float = 0.95,

        # Curriculum training config
        wm_warmup_steps: int = 500,
        wm_warmup_update_ratio: int = 5,
        llm_start_threshold: float = 0.3,

        # Stability monitoring config
        grad_norm_threshold: float = 100.0,
        loss_spike_threshold: float = 3.0,

        # Logging
        log_interval: int = 10,
        rank: int = 0,
    ):
        self.rank = rank
        self.log_interval = log_interval
        self.step_count = 0

        # Initialize modules
        self.value_normalizer = AdaptiveValueNormalizer(
            init_momentum=value_norm_init_momentum,
            final_momentum=value_norm_final_momentum,
            warmup_steps=value_norm_warmup_steps,
            clip_percentile=value_norm_clip_percentile,
        )

        self.curriculum_scheduler = CurriculumTrainingScheduler(
            wm_warmup_steps=wm_warmup_steps,
            wm_warmup_update_ratio=wm_warmup_update_ratio,
            llm_start_threshold=llm_start_threshold,
        )

        self.stability_monitor = StabilityMonitor(
            grad_norm_threshold=grad_norm_threshold,
            loss_spike_threshold=loss_spike_threshold,
        )

        logging.info(f"[Rank {rank}] PriorZero Stability Optimizer initialized")
        logging.info(f"  - WM warmup steps: {wm_warmup_steps}")
        logging.info(f"  - LLM start threshold: {llm_start_threshold}")
        logging.info(f"  - Value norm warmup: {value_norm_warmup_steps}")

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
        logging.info(f"  - Is warming up: {curriculum_stats['is_warming_up']}")
        if curriculum_stats['value_prediction_accuracy'] is not None:
            logging.info(f"  - Value prediction accuracy: {curriculum_stats['value_prediction_accuracy']:.3f}")

        logging.info("\nValue Normalization:")
        logging.info(f"  - Total updates: {value_stats.get('total_updates', 0)}")
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
