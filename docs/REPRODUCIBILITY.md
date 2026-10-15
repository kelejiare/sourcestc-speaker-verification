# 训练与运行说明

## 推理

运行环境为 Ubuntu 20.04、Python 3.8.20、PyTorch 2.0.1 和 NumPy 1.24.4。

1. 从 [权重下载页](../weights/README.md) 获取模型包，放到 `weights/sourcestc-weights.tar.gz`。
2. 按 `data_sceneryN/audio/**/*.wav` 结构准备输入。
3. 执行 `bash scripts/infer.sh`，结果写入 `outputs/submit.zip`。

```bash
XFDATA_ROOT=/path/to/sourcestc-test \
PREDICTION_RESULT_DIR="$PWD/outputs" \
bash scripts/infer.sh
```

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `MODEL_PACKAGE` | 模型压缩包路径 | `weights/sourcestc-weights.tar.gz` |
| `XFDATA_ROOT` | 输入目录 | `data/sourcestc-test` |
| `PREDICTION_RESULT_DIR` | 输出目录 | `outputs` |
| `GAME823_DEVICE` | 运行设备 | `auto`，可选 `cpu` 或 `cuda` |
| `GAME823_TORCH_THREADS` | CPU 线程数 | `4` |

脚本自动准备模型文件。推理过程中无需联网下载模型。

## 训练流程

训练代码位于 `training/source/`，入口见 [训练代码索引](../training/CODE_MAP.md)。

1. 准备多域语音数据与声纹特征缓存。
2. 训练输出空间蒸馏适配器，以及来源感知和短时长适配器。
3. 构建 ReDimNet Plus 与 w2v-BERT 特征缓存。
4. 训练 w2v-BERT 的 LoRA、说话人适配头与分类器。
5. 评估来源、时长对应的融合系数与教师特征校正策略。

训练脚本使用 `data/`、`model/` 和 `records/` 等相对目录。运行各阶段前，需要按脚本准备训练数据、预训练模型和对应缓存；这些训练资产不包含在源码仓库内。

根目录的 `requirements.txt` 用于推理。教师模型准备工具还使用 Transformers，离线训练时需另行准备对应环境。

## 输出格式

`submit.zip` 保留输入的相对目录结构，将 WAV 后缀替换为 NPZ。每个 NPZ 包含：

| 字段 | 类型 | 形状 |
| --- | --- | --- |
| `embedding` | `float32` | `(3776,)` |

嵌入向量经过 L2 归一化，可用于余弦相似度计算。
