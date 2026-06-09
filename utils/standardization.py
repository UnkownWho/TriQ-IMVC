# utils/standardization.py
"""
标准化工具函数
包含：普通标准化、使用预训练标准化器、标准化器保存与恢复
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def standardize_batch(batch_data):
    """
    对批次数据进行标准化（使用批次自身的统计）
    
    Args:
        batch_data: 批次数据列表 [V个视图，每个形状为[n, feature_dim]]
        
    Returns:
        标准化后的数据列表
    """
    standardized_data = []
    for data in batch_data:
        # 转换为numpy进行标准化
        data_np = data.cpu().numpy()
        scaler = MinMaxScaler()
        standardized_np = scaler.fit_transform(data_np)
        standardized_data.append(torch.tensor(standardized_np).to(data.device))
    
    return standardized_data


def standardize_for_training(data_list, mask, device=None):
    """
    为训练数据标准化（考虑缺失视图，使用数据自身的统计）
    
    Args:
        data_list: 数据列表 [V个视图，每个形状为[n, feature_dim]]
        mask: 掩码矩阵 [n, V]
        device: 设备
        
    Returns:
        标准化后的数据列表
    """
    if device is None:
        device = data_list[0].device
    
    v_num = len(data_list)
    n_samples = mask.shape[0]
    standardized_data = []
    
    for v in range(v_num):
        # 获取该视图的存在样本
        mask_v = mask[:, v].bool()
        
        if mask_v.any():
            # 只对存在样本进行标准化
            exist_data = data_list[v][mask_v].cpu().numpy()
            scaler = MinMaxScaler()
            standardized_exist = scaler.fit_transform(exist_data)
            
            # 创建完整的数据张量
            feature_dim = data_list[v].shape[1]
            full_data = torch.zeros((n_samples, feature_dim), device=device)
            full_data[mask_v] = torch.tensor(standardized_exist).to(device)
            standardized_data.append(full_data)
        else:
            # 该视图没有存在样本
            feature_dim = data_list[v].shape[1]
            standardized_data.append(torch.zeros((n_samples, feature_dim), device=device))
    
    return standardized_data


def standardize_single_view(data, mask=None):
    """
    标准化单个视图的数据
    
    Args:
        data: 数据 [n, feature_dim]
        mask: 掩码 [n] (可选)
        
    Returns:
        标准化后的数据
    """
    data_np = data.cpu().numpy()
    
    if mask is not None:
        # 只对存在样本标准化
        mask_np = mask.cpu().numpy()
        if mask_np.any():
            exist_data = data_np[mask_np]
            scaler = MinMaxScaler()
            standardized_exist = scaler.fit_transform(exist_data)
            
            # 创建完整数据
            standardized_data = np.zeros_like(data_np)
            standardized_data[mask_np] = standardized_exist
            return torch.tensor(standardized_data).to(data.device)
        else:
            return torch.zeros_like(data)
    else:
        # 标准化所有数据
        scaler = MinMaxScaler()
        standardized_np = scaler.fit_transform(data_np)
        return torch.tensor(standardized_np).to(data.device)


# ============ 新增：预训练标准化器相关功能 ============

def create_and_fit_scalers(data_list, mask=None):
    """
    创建并拟合标准化器
    
    Args:
        data_list: 数据列表 [V个视图]
        mask: 掩码矩阵 [n, V] (可选)，如果提供则只使用存在的样本
        
    Returns:
        拟合好的标准化器列表
    """
    v_num = len(data_list)
    scalers = []
    
    for v in range(v_num):
        scaler = MinMaxScaler()
        
        if mask is not None:
            # 只使用存在的样本
            mask_v = mask[:, v].bool()
            if mask_v.any():
                exist_data = data_list[v][mask_v].cpu().numpy()
                scaler.fit(exist_data)
            else:
                # 如果没有存在样本，拟合空数据
                scaler.fit(np.zeros((0, data_list[v].shape[1])))
        else:
            # 使用所有样本
            data_np = data_list[v].cpu().numpy()
            scaler.fit(data_np)
        
        scalers.append(scaler)
    
    return scalers


def extract_scaler_params(scalers):
    """
    从标准化器列表中提取参数
    
    Args:
        scalers: 标准化器列表
        
    Returns:
        参数列表，每个元素是标准化器参数的字典
    """
    params_list = []
    
    for scaler in scalers:
        params = {
            'data_min': scaler.data_min_,
            'data_max': scaler.data_max_,
            'min': scaler.min_,
            'scale': scaler.scale_,
            'data_range': scaler.data_range_,
            'n_features_in': scaler.n_features_in_,
        }
        
        # 保存可选属性
        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            params['feature_names_in'] = scaler.feature_names_in_
        
        params_list.append(params)
    
    return params_list


def restore_scalers_from_params(params_list):
    """
    从参数重建标准化器
    
    Args:
        params_list: 标准化器参数列表
        
    Returns:
        重建的标准化器列表
    """
    scalers = []
    
    for params in params_list:
        scaler = MinMaxScaler()
        
        # 恢复参数
        scaler.data_min_ = params['data_min']
        scaler.data_max_ = params['data_max']
        scaler.scale_ = params['scale']
        scaler.min_ = params['min']
        scaler.data_range_ = params['data_range']
        scaler.n_features_in_ = params['n_features_in']
        
        # 恢复可选属性
        if 'feature_names_in' in params:
            scaler.feature_names_in_ = params['feature_names_in']
        
        scalers.append(scaler)
    
    return scalers


def standardize_with_scalers(data_list, scalers, device=None):
    """
    使用给定的标准化器标准化数据
    
    Args:
        data_list: 数据列表 [V个视图]
        scalers: 标准化器列表 [V个]
        device: 输出设备
        
    Returns:
        标准化后的数据列表
    """
    if device is None:
        device = data_list[0].device if len(data_list) > 0 else torch.device('cpu')
    
    v_num = len(data_list)
    standardized_data = []
    
    for v in range(v_num):
        # 转换为numpy
        data_np = data_list[v].cpu().numpy()
        
        # 使用标准化器进行转换
        standardized_np = scalers[v].transform(data_np)
        
        # 转回tensor
        standardized_tensor = torch.tensor(standardized_np, device=device)
        standardized_data.append(standardized_tensor)
    
    return standardized_data


def standardize_for_training_with_scalers(data_list, mask, scalers, device=None):
    """
    使用给定的标准化器为训练数据标准化（考虑缺失视图）
    
    Args:
        data_list: 数据列表 [V个视图]
        mask: 掩码矩阵 [n, V]
        scalers: 标准化器列表 [V个]
        device: 输出设备
        
    Returns:
        标准化后的数据列表
    """
    if device is None:
        device = data_list[0].device
    
    v_num = len(data_list)
    n_samples = mask.shape[0]
    standardized_data = []
    
    for v in range(v_num):
        # 获取该视图的存在样本
        mask_v = mask[:, v].bool()
        
        if mask_v.any():
            # 只对存在样本进行标准化
            exist_data = data_list[v][mask_v].cpu().numpy()
            standardized_exist = scalers[v].transform(exist_data)
            
            # 创建完整的数据张量
            feature_dim = data_list[v].shape[1]
            full_data = torch.zeros((n_samples, feature_dim), device=device)
            full_data[mask_v] = torch.tensor(standardized_exist).to(device)
            standardized_data.append(full_data)
        else:
            # 该视图没有存在样本
            feature_dim = data_list[v].shape[1]
            standardized_data.append(torch.zeros((n_samples, feature_dim), device=device))
    
    return standardized_data


# 为方便使用，创建别名
standardize_with_pretrained_scalers = standardize_with_scalers
standardize_batch_with_scalers = standardize_with_scalers