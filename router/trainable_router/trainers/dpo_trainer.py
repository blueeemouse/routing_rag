"""
DPO (Direct Preference Optimization) Trainer

基于TRL库的DPOTrainer，适配分类任务的路由器训练。

参考: EllieSQL/src/dpo/qwen_dpo_train.py
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path

# 尝试导入TRL库
try:
    from trl import DPOTrainer
    from trl.trainer.dpo_config import DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    DPOTrainer = object
    DPOConfig = None

from transformers.trainer_callback import TrainerCallback


class SaveLogCallback(TrainerCallback):
    """
    自定义回调：保存训练日志到txt文件
    """
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        # 初始化时清空文件，写入表头
        with open(self.log_file, "w") as f:
            f.write("step, train_loss, eval_loss\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            # 获取训练损失和评估损失
            train_loss = logs.get("loss", "NA")
            eval_loss = logs.get("eval_loss", "NA")
            # 追加到日志文件
            with open(self.log_file, "a") as f:
                f.write(f"{step}, {train_loss}, {eval_loss}\n")


class ClassificationDPOTrainer(DPOTrainer):
    """
    分类任务的DPO Trainer
    
    适配TRL的DPOTrainer，用于策略分类任务（而非生成任务）。
    关键点：
    1. tokenize_row: 只tokenize prompt，chosen/rejected直接保存为label索引
    2. concatenated_forward: 计算分类logits，然后通过log_softmax得到log概率
    """
    
    def __init__(self, *args, **kwargs):
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL库未安装，无法使用DPOTrainer。"
                "请运行: pip install trl"
            )
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def tokenize_row(
        features, 
        processing_class, 
        max_prompt_length, 
        max_completion_length, 
        add_special_tokens
    ):
        """
        Tokenize prompt，并将chosen/rejected保存为label索引
        
        Args:
            features: 数据样本，包含 prompt, chosen, rejected
            processing_class: tokenizer
            max_prompt_length: 最大prompt长度
            max_completion_length: 最大completion长度（分类任务不使用，但必须提供）
            add_special_tokens: 是否添加特殊token
            
        Returns:
            tokenized数据字典
        """
        # Tokenize prompt
        prompt_encoded = processing_class(
            features["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )
        prompt_input_ids = prompt_encoded["input_ids"]
        prompt_attention_mask = prompt_encoded.get("attention_mask")
        
        # chosen和rejected是整数label（0或1），包装为单元素列表
        # collator会将其转换为tensor shape [batch, 1]
        chosen_input_ids = [int(features["chosen"])]
        rejected_input_ids = [int(features["rejected"])]
        
        result = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }
        
        if prompt_attention_mask is not None:
            result["prompt_attention_mask"] = prompt_attention_mask
            
        return result

    def concatenated_forward(self, model, batch):
        """
        前向传播步骤：
        1. 获取tokenized的prompt_input_ids和attention_mask
        2. 模型前向传播，得到logits [batch, num_labels]
        3. 计算log_softmax，得到每个类别的log概率
        4. 使用gather根据label获取对应的log概率
        5. 返回chosen_logps和rejected_logps用于DPO损失计算
        
        Args:
            model: 策略模型
            batch: 批次数据
            
        Returns:
            包含chosen_logps, rejected_logps的字典
        """
        input_ids = batch["prompt_input_ids"]  # [batch, seq_len]
        
        # 如果collator已生成attention_mask则使用，否则自动生成
        attention_mask = batch.get("prompt_attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.args.padding_value).long()
        
        # 模型前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, num_labels]
        
        # 计算log softmax概率
        logprobs = torch.log_softmax(logits, dim=-1)  # [batch, num_labels]
        
        # 将单元素列表转换为tensor [batch, 1]，然后squeeze为[batch]
        chosen_label = batch["chosen_input_ids"].squeeze(1)
        rejected_label = batch["rejected_input_ids"].squeeze(1)
        
        # 使用gather获取每个样本对应label的log概率
        chosen_logps = logprobs.gather(1, chosen_label.unsqueeze(1)).squeeze(1)
        rejected_logps = logprobs.gather(1, rejected_label.unsqueeze(1)).squeeze(1)
        
        # 返回DPO所需的logps，同时提供mean_chosen/rejected_logits满足TRL内部统计需求
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logps,  # 分类任务中用logps代替logits
            "mean_rejected_logits": rejected_logps,
        }


class RouterDPOTrainer:
    """
    路由器DPO训练器包装类
    
    提供与routing_rag项目其他训练器一致的接口，内部使用ClassificationDPOTrainer。
    """
    
    def __init__(
        self,
        model,
        ref_model,
        config,
        output_dir: str = "outputs",
        logger=None,
        **kwargs
    ):
        """
        初始化DPO训练器
        
        Args:
            model: 策略模型（Policy）
            ref_model: 参考模型（Reference），用于计算DPO的对比损失
            config: 训练配置
            output_dir: 输出目录
            logger: 日志记录器
        """
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL库未安装，无法使用DPOTrainer。"
                "请运行: pip install trl"
            )
        
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.global_step = 0
        self.epoch = 0
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部DPO trainer，在train方法中初始化
        self._dpo_trainer = None
    
    def compute_loss(self, batch) -> torch.Tensor:
        """计算损失（由ClassificationDPOTrainer处理）"""
        raise NotImplementedError("RouterDPOTrainer使用ClassificationDPOTrainer内部处理损失计算")
    
    def train_epoch(self, dataloader, **kwargs) -> Dict[str, float]:
        """训练一个epoch（由ClassificationDPOTrainer处理）"""
        raise NotImplementedError("RouterDPOTrainer使用ClassificationDPOTrainer内部处理训练循环")
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """评估模型（由ClassificationDPOTrainer处理）"""
        if self._dpo_trainer is None:
            raise RuntimeError("必须先调用train()初始化DPO trainer")
        
        eval_result = self._dpo_trainer.evaluate()
        return {
            'eval_loss': eval_result.get('eval_loss', 0.0),
        }
    
    def train(
        self, 
        train_dataset, 
        eval_dataset=None, 
        tokenizer=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集
            tokenizer: 分词器
            
        Returns:
            训练历史
        """
        # 从config中提取DPO训练参数
        training_config = self.config.training if hasattr(self.config, 'training') else self.config
        
        # 构建DPOConfig
        dpo_config = DPOConfig(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=getattr(training_config, 'batch_size', 8),
            per_device_eval_batch_size=getattr(training_config, 'batch_size', 8),
            num_train_epochs=getattr(training_config, 'epochs', 3),
            learning_rate=getattr(training_config, 'learning_rate', 1e-5),
            fp16=getattr(training_config, 'fp16', True),
            bf16=getattr(training_config, 'bf16', False),
            logging_steps=getattr(training_config, 'logging_steps', 50),
            eval_steps=getattr(training_config, 'eval_steps', 100),
            save_steps=getattr(training_config, 'save_steps', 100),
            max_length=getattr(training_config, 'max_length', 512),
            max_prompt_length=getattr(training_config, 'max_length', 512),
            max_completion_length=1,  # 分类任务不使用
            beta=getattr(training_config, 'beta', 0.1),  # DPO温度参数
            loss_type=getattr(training_config, 'loss_type', 'sigmoid'),
            generate_during_eval=False,  # 分类任务不需要生成
            remove_unused_columns=False,
            dataset_num_proc=None,
        )
        
        # 初始化ClassificationDPOTrainer
        self._dpo_trainer = ClassificationDPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        
        # 添加日志回调
        log_file = self.output_dir / "training_log.txt"
        self._dpo_trainer.add_callback(SaveLogCallback(log_file=str(log_file)))
        
        # 开始训练
        self._dpo_trainer.train()
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_metrics': [],
        }
        
        return history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        if self._dpo_trainer is not None:
            self._dpo_trainer.save_model(path)
        else:
            # 直接保存模型
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        # 由外部模型处理加载
        pass
    
    def save_final_model(self, path: str):
        """保存最终模型"""
        self.save_checkpoint(path)
    
    def close(self):
        """清理资源"""
        pass
