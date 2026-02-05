"""
训练日志系统

提供结构化的日志记录功能，支持记录训练参数、命令、环境信息和训练指标
"""

import os
import sys
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志文件保存目录
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"router_training_{timestamp}.log"
        
        # 设置日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 配置日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建日志记录器
        self.logger = logging.getLogger("router_training")
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
        
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        return logging.getLogger(f"router_training.{name}")
    
    def log_training_start(self, command: str, args: Dict[str, Any], config: Any):
        """
        记录训练开始信息
        
        Args:
            command: 完整命令行字符串
            args: 命令行参数字典
            config: 配置对象
        """
        self.logger.info("=" * 80)
        self.logger.info("Router 训练开始")
        self.logger.info("=" * 80)
        
        # 记录命令行
        self.logger.info(f"命令行: {command}")
        self.logger.info("")
        
        # 记录时间信息
        self.logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # 记录环境信息
        self._log_environment()
        
        # 记录最终参数配置
        self._log_final_parameters(args, config)
        
    def _log_environment(self):
        """记录环境信息"""
        self.logger.info("=" * 80)
        self.logger.info("环境信息")
        self.logger.info("=" * 80)
        
        try:
            # Python 版本
            self.logger.info(f"Python 版本: {sys.version}")
            
            # PyTorch 版本
            import torch
            self.logger.info(f"PyTorch 版本: {torch.__version__}")
            
            # CUDA 信息
            if torch.cuda.is_available():
                self.logger.info(f"CUDA 可用: 是")
                self.logger.info(f"CUDA 版本: {torch.version.cuda}")
                self.logger.info(f"GPU 数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                self.logger.info(f"CUDA 可用: 否")
                
            # 操作系统
            self.logger.info(f"操作系统: {platform.system()} {platform.release()}")
            
            # Git 信息（如果有）
            try:
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                   stderr=subprocess.DEVNULL).decode('utf-8').strip()
                git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                                   stderr=subprocess.DEVNULL).decode('utf-8').strip()
                self.logger.info(f"Git 分支: {git_branch}")
                self.logger.info(f"Git Commit: {git_commit[:8]}")
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"获取环境信息时出错: {e}")
            
        self.logger.info("")
    
    def _log_final_parameters(self, args: Dict[str, Any], config: Any):
        """记录最终参数配置（合并后的）"""
        self.logger.info("=" * 80)
        self.logger.info("最终参数配置")
        self.logger.info("=" * 80)
        
        # 识别命令行覆盖的参数（非None且不是特殊参数）
        overridden_params = {}
        for key, value in sorted(args.items()):
            if value is not None and key not in ['config', 'script_path', 'resume', 'use_amp']:
                overridden_params[key] = value
        
        # 记录命令行覆盖的参数
        if overridden_params:
            self.logger.info("命令行覆盖参数:")
            for key, value in overridden_params.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("")
        
        # 记录完整配置
        self.logger.info("完整配置:")
        self._log_config_recursive(config, indent=2)
        self.logger.info("")
    
    def _log_script_content(self, script_path: str):
        """记录脚本文件内容"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            self.logger.info("=" * 80)
            self.logger.info(f"PowerShell脚本内容: {script_path}")
            self.logger.info("=" * 80)
            
            # 记录完整脚本内容
            lines = script_content.split('\n')
            for i, line in enumerate(lines, 1):
                self.logger.info(f"{i:4d} | {line}")
            
            self.logger.info("")
        except Exception as e:
            self.logger.warning(f"读取脚本文件失败: {e}")
    
    def _log_config_recursive(self, obj: Any, indent: int = 2):
        """递归记录配置对象"""
        prefix = " " * indent
        
        if hasattr(obj, '__dict__'):
            # 对象有 __dict__ 属性
            for key, value in sorted(obj.__dict__.items()):
                if key.startswith('_'):
                    continue  # 跳过私有属性
                if isinstance(value, (str, int, float, bool)):
                    self.logger.info(f"{prefix}{key}: {value}")
                elif isinstance(value, (list, tuple)):
                    self.logger.info(f"{prefix}{key}: {value}")
                elif isinstance(value, dict):
                    self.logger.info(f"{prefix}{key}:")
                    for k, v in sorted(value.items()):
                        self.logger.info(f"{prefix}  {k}: {v}")
                else:
                    # 递归处理复杂对象
                    self.logger.info(f"{prefix}{key}:")
                    self._log_config_recursive(value, indent + 2)
        elif isinstance(obj, dict):
            # 字典类型
            for key, value in sorted(obj.items()):
                if isinstance(value, (str, int, float, bool)):
                    self.logger.info(f"{prefix}{key}: {value}")
                else:
                    self.logger.info(f"{prefix}{key}:")
                    self._log_config_recursive(value, indent + 2)
        else:
            # 其他类型，直接转换为字符串
            self.logger.info(f"{prefix}{obj}")
    
    def log_data_info(self, train_samples: int, val_samples: int = 0, 
                     strategy_stats: Optional[Dict[str, int]] = None):
        """
        记录数据信息
        
        Args:
            train_samples: 训练样本数
            val_samples: 验证样本数
            strategy_stats: 策略覆盖统计
        """
        self.logger.info("=" * 80)
        self.logger.info("数据信息")
        self.logger.info("=" * 80)
        
        self.logger.info(f"训练样本数: {train_samples}")
        if val_samples > 0:
            self.logger.info(f"验证样本数: {val_samples}")
        
        if strategy_stats:
            self.logger.info("策略覆盖:")
            for strategy, count in sorted(strategy_stats.items()):
                self.logger.info(f"  {strategy}: {count} 样本")
        
        self.logger.info("")
    
    def log_training_step(self, epoch: int, step: int, total_steps: int, 
                         loss: float, learning_rate: Optional[float] = None):
        """
        记录训练步骤信息
        
        Args:
            epoch: 当前轮数
            step: 当前步数
            total_steps: 总步数
            loss: 损失值
            learning_rate: 学习率（可选）
        """
        if step % 100 == 0:  # 每100步记录一次
            if learning_rate is not None:
                self.logger.info(
                    f"Epoch {epoch+1} | Step {step}/{total_steps} | "
                    f"Loss: {loss:.4f} | LR: {learning_rate:.2e}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch+1} | Step {step}/{total_steps} | "
                    f"Loss: {loss:.4f}"
                )
    
    def log_evaluation(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录评估结果
        
        Args:
            metrics: 评估指标字典
            step: 步数（可选）
        """
        self.logger.info("=" * 80)
        self.logger.info("评估结果" + (f" (Step {step})" if step else ""))
        self.logger.info("=" * 80)
        
        if 'accuracy' in metrics:
            self.logger.info(f"准确率: {metrics['accuracy']:.4f}")
        
        if 'loss' in metrics:
            self.logger.info(f"损失: {metrics['loss']:.4f}")
        
        if 'strategy_accuracy' in metrics:
            self.logger.info("各策略准确率:")
            for strategy, acc in sorted(metrics['strategy_accuracy'].items()):
                self.logger.info(f"  {strategy}: {acc:.4f}")
        
        if 'num_samples' in metrics:
            self.logger.info(f"评估样本数: {metrics['num_samples']}")
        
        self.logger.info("")
    
    def log_model_save(self, path: str, step: Optional[int] = None):
        """
        记录模型保存
        
        Args:
            path: 保存路径
            step: 步数（可选）
        """
        if step:
            self.logger.info(f"模型已保存到: {path} (Step {step})")
        else:
            self.logger.info(f"模型已保存到: {path}")
    
    def log_training_end(self, total_epochs: int, total_steps: int):
        """
        记录训练结束信息
        
        Args:
            total_epochs: 总训练轮数
            total_steps: 总训练步数
        """
        self.logger.info("=" * 80)
        self.logger.info("训练完成")
        self.logger.info("=" * 80)
        self.logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总轮数: {total_epochs}")
        self.logger.info(f"总步数: {total_steps}")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info("=" * 80)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> TrainingLogger:
    """
    设置并获取训练日志记录器
    
    Args:
        log_dir: 日志文件保存目录
        log_level: 日志级别
        
    Returns:
        TrainingLogger 实例
    """
    return TrainingLogger(log_dir, log_level)


# 全局日志实例
_logger_instance: Optional[TrainingLogger] = None


def get_logger(name: str = "router_training") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger 实例
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = TrainingLogger()
    
    return _logger_instance.get_logger(name)


def init_logging(log_dir: str = "logs", log_level: str = "INFO") -> TrainingLogger:
    """
    初始化全局日志系统
    
    Args:
        log_dir: 日志文件保存目录
        log_level: 日志级别
        
    Returns:
        TrainingLogger 实例
    """
    global _logger_instance
    
    _logger_instance = TrainingLogger(log_dir, log_level)
    return _logger_instance
