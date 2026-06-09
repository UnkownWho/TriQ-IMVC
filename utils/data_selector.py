# utils/data_selector.py
"""
数据质量选择器
评估refine数据质量并选择高质量样本
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from utils.graph_adjacency import get_adjacency
from utils.util import expand_representation_to_batch
#from models.TriQIMVC import get_representations_for_selection

DEFAULT_W_VR = 0.3
DEFAULT_W_CV = 0.35
DEFAULT_W_RC = 0.35


class DataQualitySelector:
    """
    数据质量选择器
    使用预训练模型评估refine数据质量
    """
    
    def __init__(self, model, config, threshold=0.7):
        """
        Args:
            model: 预训练的TriQ-IMVC模型
            config: 配置文件
            threshold: 选择阈值θ
        """
        self.model = model
        self.config = config
        self.device = config["device"]
        self.threshold = threshold
        self.refine_topk = config.get("refine_topk", config.get("topk", 10))
        self.w_vr = config.get("w_vr", DEFAULT_W_VR)
        self.w_cv = config.get("w_cv", DEFAULT_W_CV)
        self.w_rc = config.get("w_rc", DEFAULT_W_RC)
        print(f"refine_topk:{self.refine_topk},vr:{self.w_vr},cv:{self.w_cv},rc:{self.w_rc}")

        
        # 设置模型为评估模式
        self.model.eval()
    
    def _build_batch_graphs(self, batch_data, batch_mask):
        """为批次数据构建各视图的图"""
        batch_size = batch_mask.shape[0]
        V = len(batch_data)
        
        adj_list = []
        
        for v in range(V):
            # 获取该视图存在的样本
            mask_v = batch_mask[:, v]
            exist_indices = torch.where(mask_v == 1)[0].cpu().numpy()
            
            if len(exist_indices) == 0:
                # 该视图没有存在样本,构建空的图
                adj_v = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.zeros(0, device=self.device),
                    (batch_size, batch_size),
                    device=self.device
                )
            else:
                # 获取存在的特征（转换为numpy用于图构建）
                features_v = batch_data[v][exist_indices].cpu().numpy()
                n_exist = len(exist_indices)
                
                # 调整topk：不能超过存在样本数-1
                adjusted_topk = min(self.refine_topk, max(1, n_exist - 1))
                
                # 构建该视图存在样本之间的图
                adj_exist, _ = get_adjacency(
                    features_v, 
                    n_exist, 
                    adjusted_topk,
                    method='heat'
                )
                
                # 将邻接矩阵转换为密集张量以便处理
                adj_dense = adj_exist.to_dense()
                
                # 创建批次维度的邻接矩阵
                adj_batch = torch.zeros((batch_size, batch_size), device=self.device)
                
                # 将存在样本之间的邻接关系放入正确位置
                for i, idx_i in enumerate(exist_indices):
                    for j, idx_j in enumerate(exist_indices):
                        adj_batch[idx_i, idx_j] = adj_dense[i, j]
                
                # 转换为稀疏张量
                indices = torch.nonzero(adj_batch).t()
                values = adj_batch[indices[0], indices[1]]
                adj_v = torch.sparse_coo_tensor(indices, values, adj_batch.size(), device=self.device)
            
            adj_list.append(adj_v)
        
        return adj_list
    
    def compute_quality_scores(self, batch_data, batch_mask):
        """计算批次数据中每个样本的质量得分"""
        # 确保模型处于评估模式且不计算梯度
        self.model.eval()
        
        # 核心：必须使用 torch.no_grad() 来禁用所有梯度计算
        with torch.no_grad():
            batch_size = batch_mask.shape[0]
            V = len(batch_data)

            # 1. 为每个视图提取存在样本并构建邻接矩阵
            x_list = []  # 每个视图的存在样本
            adj_list = []  # 每个视图的存在样本邻接矩阵

            for v in range(V):
                # 获取该视图存在的样本
                mask_v = batch_mask[:, v]
                exist_indices = torch.where(mask_v == 1)[0]

                if len(exist_indices) == 0:
                    # 该视图没有存在样本
                    feature_dim = batch_data[v].shape[1]
                    x_v = torch.zeros((0, feature_dim), device=self.device)
                    adj_v = torch.sparse_coo_tensor(
                        torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        torch.zeros(0, device=self.device),
                        (0, 0),
                        device=self.device
                    )
                else:
                    # 提取存在样本的数据
                    x_v = batch_data[v][exist_indices]

                    # 获取存在样本的特征（转换为numpy用于图构建）
                    features_v = x_v.cpu().numpy()
                    n_exist = len(exist_indices)

                    # 调整topk：不能超过存在样本数-1
                    adjusted_topk = min(self.refine_topk, max(1, n_exist - 1))

                    # 构建该视图存在样本之间的图
                    adj_v, _ = get_adjacency(
                        features_v,
                        n_exist,
                        adjusted_topk,
                        method='heat'
                    )
                    adj_v = adj_v.to(self.device)

                x_list.append(x_v)
                adj_list.append(adj_v)

            # 2. 获取模型中间表示 - 整个操作在 torch.no_grad() 中
            view_weights, encodings, reconstructions = self.model.get_representations_for_selection(
                x_list, adj_list, batch_mask
            )

            # 3. 计算视图丰富度
            vr_scores = self._compute_view_richness(batch_mask, view_weights)

            # 4. 计算编码相似度
            cv_scores = self._compute_cross_view_similarity(encodings, batch_mask, batch_size)

            # 5. 计算重构误差
            rc_scores = self._compute_reconstruction_score(batch_data, reconstructions, batch_mask, batch_size)

            # 6. 加权综合得分
            total_scores = self.w_vr * vr_scores + self.w_cv * cv_scores + self.w_rc * rc_scores

        return total_scores, (vr_scores, cv_scores, rc_scores)
    
    def _compute_view_richness(self, batch_mask, view_weights):
        """计算视图丰富度得分"""
        scores = []
        for i in range(batch_mask.shape[0]):
            mask_i = batch_mask[i]
            numerator = torch.sum(mask_i * view_weights)
            denominator = torch.sum(view_weights)
            score = numerator / denominator if denominator > 0 else 0
            scores.append(score)
        return torch.stack(scores)
    
    def _compute_cross_view_similarity(self, encodings, batch_mask, batch_size):
        """计算不同视图编码的相似度"""
        scores = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # 找出该样本存在的视图
            existing_views = torch.where(batch_mask[i] == 1)[0]
            
            if len(existing_views) < 2:
                scores[i] = 1.0  # 只有一个视图，无法计算相似度
                continue
            
            # 获取该样本在这些视图上的编码
            sample_encodings = []
            for v in existing_views:
                # 找到该样本在视图v中的位置
                mask_v = batch_mask[:, v]
                exist_idx = torch.where(mask_v == 1)[0]
                idx_in_exist = torch.where(exist_idx == i)[0]
                
                if len(idx_in_exist) > 0:
                    encoding = encodings[v][idx_in_exist[0]]
                    sample_encodings.append(encoding)
            
            if len(sample_encodings) < 2:
                scores[i] = 1.0
                continue
            
            # 计算所有视图对之间的余弦相似度
            similarities = []
            for a in range(len(sample_encodings)):
                for b in range(a + 1, len(sample_encodings)):
                    sim = F.cosine_similarity(
                        sample_encodings[a].unsqueeze(0),
                        sample_encodings[b].unsqueeze(0)
                    )
                    similarities.append(sim)
            
            if len(similarities) > 0:
                scores[i] = torch.mean(torch.stack(similarities))
            else:
                scores[i] = 1.0
        
        return scores
    
    def _compute_reconstruction_score(self, batch_data, reconstructions, batch_mask, batch_size):
        """计算重构误差得分"""
        scores = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            total_mse = 0
            count = 0
            
            for v in range(len(batch_data)):
                if batch_mask[i, v] == 1:
                    # 找到该样本在视图v中的索引
                    mask_v = batch_mask[:, v]
                    exist_idx = torch.where(mask_v == 1)[0]
                    idx_in_exist = torch.where(exist_idx == i)[0]
                    
                    if len(idx_in_exist) > 0:
                        original = batch_data[v][i]
                        recon = reconstructions[v][idx_in_exist[0]]
                        
                        # 计算MSE
                        mse = F.mse_loss(original, recon)
                        total_mse += mse
                        count += 1
            
            if count > 0:
                avg_mse = total_mse / count
                # 将MSE转换为得分：exp(-mse)
                scores[i] = torch.exp(-avg_mse)
            else:
                scores[i] = 0  # 所有视图都缺失
        
        return scores
    
    def select_quality_data(self, batch_data, batch_mask, batch_idx, save_dir=None):
        """
        选择高质量数据
        
        Args:
            batch_data: 批次数据
            batch_mask: 批次掩码
            batch_idx: 批次索引
            save_dir: 保存结果的目录
            
        Returns:
            selected_indices: 选中样本的索引（相对于批次）
            total_scores: 所有样本的质量分数
        """
        # 计算质量得分
        total_scores, component_scores = self.compute_quality_scores(batch_data, batch_mask)
        
        # 根据阈值选择
        selected_mask = total_scores >= self.threshold
        selected_indices = torch.where(selected_mask)[0].cpu().numpy()
        
        # 统计信息
        n_selected = len(selected_indices)
        n_total = batch_mask.shape[0]
        selection_rate = n_selected / n_total if n_total > 0 else 0
        
        print(f"批次 {batch_idx}: 选中 {n_selected}/{n_total} ({selection_rate:.1%})")
        
        # 保存结果
        if save_dir is not None and n_selected > 0:
            self._save_batch_results(
                batch_idx, batch_data, batch_mask, 
                selected_indices, total_scores, component_scores, save_dir
            )
        
        return selected_indices, total_scores.cpu().numpy()
    
    def select_quality_data_with_standardization(self, batch_data, batch_mask, batch_idx, save_dir=None):
        """
        选择高质量数据（内部进行临时标准化）
        
        Args:
            batch_data: 未标准化的批次数据
            batch_mask: 批次掩码
            batch_idx: 批次索引
            save_dir: 保存结果的目录
            
        Returns:
            selected_indices: 选中样本的索引
            total_scores: 所有样本的质量分数
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # 临时标准化批次数据
        standardized_data = []
        for v in range(len(batch_data)):
            data_np = batch_data[v].cpu().numpy()
            scaler = MinMaxScaler()
            standardized_np = scaler.fit_transform(data_np)
            standardized_data.append(torch.tensor(standardized_np).to(self.device))
        
        # 使用标准化后的数据进行质量评估
        return self.select_quality_data(standardized_data, batch_mask, batch_idx, save_dir)
    
    def _save_batch_results(self, batch_idx, batch_data, batch_mask, 
                           selected_indices, total_scores, component_scores, save_dir):
        """保存批次选择结果"""
        batch_dir = os.path.join(save_dir, "selected_batches")
        os.makedirs(batch_dir, exist_ok=True)
        
        save_path = os.path.join(batch_dir, f"batch_{batch_idx:03d}.npz")
        
        data_dict = {
            'batch_idx': batch_idx,
            'selected_indices': selected_indices,
            'total_scores': total_scores.cpu().numpy(),
            'vr_scores': component_scores[0].cpu().numpy(),
            'cv_scores': component_scores[1].cpu().numpy(),
            'rc_scores': component_scores[2].cpu().numpy(),
            'batch_mask': batch_mask.cpu().numpy(),
        }
        
        # 保存每个视图的数据
        for v in range(len(batch_data)):
            data_dict[f'view_{v}_data'] = batch_data[v].cpu().numpy()
            if len(selected_indices) > 0:
                data_dict[f'view_{v}_selected'] = batch_data[v][selected_indices].cpu().numpy()
        
        np.savez(save_path, **data_dict)
