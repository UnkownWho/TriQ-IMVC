# utils/refine_loader.py
import numpy as np
import torch


class RefineBatchGenerator:
    """
    refine批次数据生成器
    使用预生成的掩码，生成固定大小的批次
    """
    
    def __init__(self, data_list, labels, mask, batch_size=100, 
                 shuffle=True, seed=42, device=None):
        """
        Args:
            data_list: 列表，每个视图的数据 [N, feature_dim]
            labels: 标签 [N]
            mask: 预生成的掩码 [N, V]
            batch_size: 批次大小
            shuffle: 是否打乱批次顺序
            seed: 随机种子
            device: 设备
        """
        self.data_list = data_list
        self.labels = torch.tensor(labels) if not torch.is_tensor(labels) else labels
        self.mask = torch.tensor(mask) if not torch.is_tensor(mask) else mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        
        self.n_samples = len(labels)
        self.v_num = len(data_list)
        
        # 准备批次索引
        self._prepare_batches()
    
    def _prepare_batches(self):
        """准备批次索引"""
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        
        # 划分批次
        self.batch_indices = []
        n_batches = int(np.ceil(self.n_samples / self.batch_size))
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.n_samples)
            batch_idx = indices[start_idx:end_idx]
            self.batch_indices.append(batch_idx)
    
    def __len__(self):
        """返回批次数量"""
        return len(self.batch_indices)
    
    def __iter__(self):
        """迭代生成批次"""
        for batch_idx in self.batch_indices:
            # 提取批次数据
            batch_data = []
            for v in range(self.v_num):
                data_v = self.data_list[v][batch_idx]
                if self.device is not None:
                    data_v = data_v.to(self.device)
                batch_data.append(data_v)
            
            # 提取批次标签和掩码
            batch_labels = self.labels[batch_idx]
            batch_mask = self.mask[batch_idx]
            
            if self.device is not None:
                batch_labels = batch_labels.to(self.device)
                batch_mask = batch_mask.to(self.device)
            
            yield batch_data, batch_labels, batch_mask, batch_idx
    
    def reset(self):
        """重置生成器，重新打乱批次顺序"""
        self._prepare_batches()
    
    def get_full_data(self):
        """获取完整数据（用于调试）"""
        return self.data_list, self.labels, self.mask

def create_refine_batches(config, data_list, labels, mask):
    """
    加载refine学习数据
    
    Args:
        config: 配置文件
        data_list: 完整数据列表
        labels: 完整标签
        mask: 完整掩码
        
    Returns:
        refine_generator: refine批次生成器
    """
    # 直接从config获取refine参数
    # 注意：这里假设config中已经有这些参数，如果没有则使用默认值
    
    # 获取批次大小，如果不存在则使用默认值
    batch_size = config.get("refine_batch_size", 100)  # 默认批次大小100
    
    # 获取其他参数
    shuffle = config.get("shuffle_batches", True)  # 默认打乱批次
    seed = config.get("refine_seed", config.get("seed", 42))  # 使用配置中的seed或默认值
    device = config.get("device", torch.device("cpu"))  # 设备
    
    # 创建refine批次生成器
    generator = RefineBatchGenerator(
        data_list=data_list,
        labels=labels,
        mask=mask,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        device=device
    )
    
    print(f"创建refine批次生成器: {len(generator)} 个批次, "
          f"每个批次 {batch_size} 个样本")
    
    return generator
