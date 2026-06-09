import math
import os

import numpy as np
from numpy.random import randint
import torch
import logging
import random

from utils.paths import LOG_DIR, ensure_dir


def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()


def setup_seed(seed_n):
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def format_device_info(device):
    """Return a concise description of the runtime device."""
    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        return f"Running on GPU: cuda:{gpu_index} ({gpu_name})"
    return "Running on CPU"


def write_eva(filepath, ms, *arg):

    with open(filepath, 'a') as file:

        file.write('missing-rate:{:.1f} \t '.format(ms))
        output = 'ACC_mean:'+str(round(np.mean(arg[0]) * 100, 2)) + ',  ' + 'ACC_std:'+str(round(np.std(arg[0]) * 100, 2)) + ';  ' + \
                'NMI_mean:'+str(round(np.mean(arg[1]) * 100, 2)) + ',  ' + 'NMI_std:'+str(round(np.std(arg[1]) * 100, 2)) + ';  ' + \
                'ARI_mean:'+str(round(np.mean(arg[2]) * 100, 2)) + ',  ' + 'ARI_std:'+str(round(np.std(arg[2]) * 100, 2)) + ';\n'
        file.write(output)
        file.flush()


def get_mask( data_size, missing_ratio,view_num):
    """
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size*missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask


def get_logger(config, main_dir=None, prefix=None):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Reinitialize handlers so repeated runs don't duplicate log output.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    log_dir = ensure_dir(main_dir or LOG_DIR)
    name_parts = []
    if prefix:
        name_parts.append(str(prefix))
    name_parts.append(str(config['dataset']))
    name_parts.append(str(config['missing_rate']).replace('.', '_'))
    log_path = log_dir / ("_".join(name_parts) + ".logs")

    fh = logging.FileHandler(log_path)

    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def cal_std(logger, *arg):
    """ print clustering results """
    if len(arg) == 3:
        logger.info("ACC"+str(arg[0]))
        logger.info("NMI"+str(arg[1]))
        logger.info("ARI"+str(arg[2]))
        output = """ 
                     ACC {:.2f} std {:.2f}
                     NMI {:.2f} std {:.2f} 
                     ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100, np.std(arg[0]) * 100, np.mean(arg[1]) * 100,
                                                     np.std(arg[1]) * 100, np.mean(arg[2]) * 100, np.std(arg[2]) * 100)
        logger.info(output)
        output2 = str(round(np.mean(arg[0]) * 100, 2)) + ',' + str(round(np.std(arg[0]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[1]) * 100, 2)) + ',' + str(round(np.std(arg[1]) * 100, 2)) + ';' + \
                  str(round(np.mean(arg[2]) * 100, 2)) + ',' + str(round(np.std(arg[2]) * 100, 2)) + ';\n'
        logger.info(output2)
        return round(np.mean(arg[0]) * 100, 2), round(np.mean(arg[1]) * 100, 2), round(np.mean(arg[2]) * 100, 2)

    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
        logger.info(output)

def stratified_split_with_mask(data_list, labels, mask, train_ratio=0.6, seed=42):
    """
    对多视图数据、标签和掩码进行分层分割
    保证每个类别的样本在预训练和refine部分都有相同比例
    
    Args:
        data_list: 列表，每个视图的数据 [N, feature_dim]
        labels: 标签 [N]
        mask: 掩码矩阵 [N, V]
        train_ratio: 预训练比例
        seed: 随机种子
        
    Returns:
        train_data, train_labels, train_mask, refine_data, refine_labels, refine_mask
    """
    from sklearn.model_selection import train_test_split
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # 分层分割索引
    train_idx, refine_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # 分割数据
    train_data = [data[train_idx] for data in data_list]
    refine_data = [data[refine_idx] for data in data_list]
    
    # 分割标签和掩码
    train_labels = labels[train_idx]
    refine_labels = labels[refine_idx]
    train_mask = mask[train_idx]
    refine_mask = mask[refine_idx]
    
    return train_data, train_labels, train_mask, refine_data, refine_labels, refine_mask


def expand_representation_to_batch(encoding, mask_vector, batch_size, feature_dim, device, fill_value=0):
    """
    将视图特定编码扩展回批次维度
    
    Args:
        encoding: 存在样本的编码 [n_exist, feature_dim]
        mask_vector: 该视图的存在掩码 [batch_size]
        batch_size: 批次大小
        feature_dim: 特征维度
        device: 设备
        fill_value: 缺失位置的填充值
        
    Returns:
        扩展后的编码 [batch_size, feature_dim]
    """
    expanded = torch.full((batch_size, feature_dim), fill_value, device=device)
    exist_indices = torch.where(mask_vector == 1)[0]
    
    if len(exist_indices) > 0:
        expanded[exist_indices] = encoding
    
    return expanded
    
