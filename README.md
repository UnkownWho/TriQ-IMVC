# PR-TriQ-IMVC 运行说明

## 数据放置

将数据集 `.mat` 文件放到 `data/` 目录：

## 单次预训练

```bash
python main/train.py --dataset 100leaves --missing_rate 0.5 --seed 393
```

## 批量预训练

```bash
python main/multi_train.py --dataset 100leaves --missing_rate 0.5
```

## 单模型 refine/boost

先完成预训练，再把生成的 `.pth` 路径传入：

```bash
python main/boost.py --model_path outputs/pretrain/100leaves_missing0.5_mask1_split42_seed393.pth
```

## 批量 refine/boost

```bash
python main/multi_boost.py outputs/pretrain/model_1.pth outputs/pretrain/model_2.pth
```

也可以只传模型文件名，脚本会在 `outputs/pretrain/` 下查找：

```bash
python main/multi_boost.py 100leaves_missing0.5_mask1_split42_seed396.pth
```

## 输出目录

```text
outputs/pretrain/   # 预训练模型
outputs/refine/     # refine 后模型
outputs/logs/       # 日志
```
