# utils/memory_manager.py
"""
内存管理器
存储标准化后的数据
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import warnings
from collections import deque


class MemoryManager:
    """
    内存管理器 - 存储标准化后的数据
    """
    
    def __init__(self, feature_dims: List[int], max_capacity: Optional[int] = None):
        """
        初始化内存管理器
        
        Args:
            feature_dims: 每个视图的特征维度列表
            max_capacity: 最大容量，None表示无限制
        """
        self.feature_dims = feature_dims
        self.v_num = len(feature_dims)
        self.max_capacity = max_capacity
        
        # 使用双端队列实现FIFO淘汰策略
        self.samples = deque(maxlen=max_capacity) if max_capacity else []
        
        # 统计信息
        self.stats = {
            'total_added': 0,
            'pre_train_samples': 0,
            'refine_selected_samples': 0,
            'evicted_samples': 0,
            'current_capacity': 0
        }
    
    def add_samples(self, data_list: List[torch.Tensor], mask: torch.Tensor,
                labels: torch.Tensor, sample_type: str = 'refine',
                indices: Optional[List[int]] = None):
        """
        添加样本到内存（通用方法）

        Args:
            data_list: 标准化后的数据列表 [V个视图]
            mask: 掩码矩阵 [n_samples, V]
            labels: 标签 [n_samples]
            sample_type: 样本类型 ('pretrain' 或 'refine')
            indices: 原始索引（可选）
        """
        n_samples = mask.shape[0]

        for i in range(n_samples):
            # 创建稀疏表示
            sample = {
                'features': {},
                'mask': mask[i].tolist(),
                'label': labels[i].item() if torch.is_tensor(labels[i]) else labels[i],
                'type': sample_type
            }

            if indices is not None:
                sample['original_idx'] = indices[i]

            # 只存储存在的视图特征（标准化后的），并确保没有梯度
            for v in range(self.v_num):
                if mask[i, v] == 1:
                    # 关键：确保没有梯度历史
                    feat = data_list[v][i]
                    if torch.is_tensor(feat) and feat.requires_grad:
                        # 创建完全独立的副本
                        feat = feat.detach().clone().cpu()
                    elif torch.is_tensor(feat):
                        feat = feat.clone().cpu()
                    else:
                        feat = feat.cpu() if hasattr(feat, 'cpu') else feat
                    
                    sample['features'][v] = feat

            # 添加到内存
            self.samples.append(sample)

        # 更新统计
        self.stats['total_added'] += n_samples
        self.stats['current_capacity'] = len(self.samples)

        if sample_type == 'pretrain':
            self.stats['pre_train_samples'] += n_samples
        else:
            self.stats['refine_selected_samples'] += n_samples
    
    def initialize_with_pretrain(self, data_list: List[torch.Tensor], mask: torch.Tensor,
                                labels: torch.Tensor, indices: Optional[List[int]] = None):
        """
        用预训练数据初始化内存
        """
        print(f"用预训练数据初始化内存: {mask.shape[0]} 个样本")
        self.add_samples(data_list, mask, labels, 'pretrain', indices)
        print(f"内存初始化完成: {self.stats['current_capacity']} 个样本")
    
    def add_refine_samples(self, batch_data: List[torch.Tensor], batch_mask: torch.Tensor,
                               batch_labels: torch.Tensor, selected_indices: List[int],
                               original_indices: Optional[List[int]] = None):
        """
        添加refine选择的样本到内存
        """
        n_selected = len(selected_indices)
        print(f"添加 {n_selected} 个refine样本到内存")
        
        # 提取选中的数据
        selected_data = [data[selected_indices] for data in batch_data]
        selected_mask = batch_mask[selected_indices]
        selected_labels = batch_labels[selected_indices]
        
        self.add_samples(selected_data, selected_mask, selected_labels, 'refine', original_indices)
    
    def random_sample(self, n: int, device: Optional[torch.device] = None) -> Tuple:
        """ 
        从内存中随机抽取n个样本
        """
        if n > len(self.samples):
            n = len(self.samples)
            print(f"警告: 请求抽取 {n} 个样本，但内存中只有 {len(self.samples)} 个")
        
        if n == 0:
            return self._empty_output()
        
        # 随机选择
        indices = random.sample(range(len(self.samples)), n)
        selected = [self.samples[i] for i in indices]
        
        return self._to_dense(selected, device)
    
    def _empty_output(self):
        """返回空数据"""
        data_list = [torch.zeros((0, dim)) for dim in self.feature_dims]
        mask = torch.zeros((0, self.v_num))
        labels = torch.zeros(0)
        return data_list, mask, labels
    
    def _to_dense(self, samples: List[Dict], device: Optional[torch.device] = None):
        """将稀疏样本转换为密集表示"""
        n_samples = len(samples)

        # 初始化
        data_list = [torch.zeros((n_samples, dim)) for dim in self.feature_dims]
        mask = torch.zeros((n_samples, self.v_num))
        labels = []

        # 填充
        for i, sample in enumerate(samples):
            mask[i] = torch.tensor(sample['mask'])

            for v, feat in sample['features'].items():
                # 确保特征数据是张量且没有梯度
                if torch.is_tensor(feat):
                    # 如果已经在设备上，先移到CPU再处理
                    feat_cpu = feat.cpu() if feat.is_cuda else feat
                    # 创建没有梯度的新张量
                    feat_clean = feat_cpu.detach().clone()
                else:
                    # 如果是numpy数组或其他类型，转换为张量
                    feat_clean = torch.tensor(feat)
                
                data_list[v][i] = feat_clean

            labels.append(sample['label'])

        labels = torch.tensor(labels)

        # 转移到设备
        if device is not None:
            data_list = [data.to(device) for data in data_list]
            mask = mask.to(device)
            labels = labels.to(device)

        return data_list, mask, labels
    
    def get_all_data(self, device: Optional[torch.device] = None):
        """获取所有数据"""
        return self._to_dense(list(self.samples), device)
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n内存统计:")
        print(f"  当前容量: {self.stats['current_capacity']}")
        print(f"  预训练样本: {self.stats['pre_train_samples']}")
        print(f"  refine样本: {self.stats['refine_selected_samples']}")
        print(f"  总添加: {self.stats['total_added']}")
        if self.max_capacity:
            print(f"  最大容量: {self.max_capacity}")
            print(f"  使用率: {self.stats['current_capacity']/self.max_capacity:.1%}")
