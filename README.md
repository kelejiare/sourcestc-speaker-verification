# SourceSTC 复杂场景说话人确认

2026 科大讯飞 AI 开发者大赛“声纹迷雾复杂场景说话人确认挑战赛”参赛方案。

项目面向移动录音、会议录音和短语音场景，结合多主干声纹表征、短语音适配与教师特征融合，输出用于余弦匹配的说话人嵌入。

## 方法

- **多主干表征**：融合 ResNet、CAM++ 和 ReDimNet 的声纹特征。
- **短语音适配**：通过多域蒸馏、来源感知适配器和度量学习处理时长与录音条件差异。
- **教师特征融合**：使用 w2v-BERT 与 LoRA 提取补充特征，按来源和时长设置融合系数。
- **特征校正**：按来源和时长分组，对教师特征进行无标签均值校正。

输出为 3776 维、L2 归一化的 `float32` 向量。每条 WAV 对应一个 NPZ 文件，字段名为 `embedding`，结果打包为 `submit.zip`。

## 比赛成绩

| 阶段 | 分数 |
| --- | ---: |
| 初赛 | 92.37968 |
| 复赛 | 93.15870 |

两阶段的评测集合不同，成绩分别报告。

## 权重下载

[百度网盘下载](https://pan.baidu.com/s/1NMlsouDgTRTS9uIV47wl1w?pwd=v464)，提取码：`v464`，约 1.51 GB。模型包仅包含权重文件，使用方式见 [权重说明](weights/README.md)。

下载后将压缩包命名为 `sourcestc-weights.tar.gz`，放入 `weights/`，无需手动解压。

## 环境与运行

- Ubuntu 20.04
- Python 3.8.20
- PyTorch 2.0.1
- NumPy 1.24.4

```bash
python3 -m pip install -r requirements.txt

XFDATA_ROOT=/path/to/sourcestc-test \
PREDICTION_RESULT_DIR="$PWD/outputs" \
bash scripts/infer.sh
```

输入目录：

```text
sourcestc-test/
├── data_scenery1/audio/*.wav
└── data_scenery2/audio/*.wav
```

结果保存在 `outputs/submit.zip`。使用 CPU 时设置 `GAME823_DEVICE=cpu`；权重位于其他位置时，用 `MODEL_PACKAGE=/path/to/package.tar.gz` 指定。

## 目录

```text
inference/       模型与推理代码
training/        训练、适配器导出与评估代码
scripts/         权重准备与推理入口
tests/           运行协议测试
weights/         权重下载说明
docs/            方法、训练与运行说明
requirements.txt
```

## 文档

- [技术方案](docs/METHOD.md)
- [训练与运行说明](docs/REPRODUCIBILITY.md)
- [成绩与测试](docs/RESULTS.md)
- [训练代码索引](training/CODE_MAP.md)

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 第三方代码

项目使用了 3D-Speaker、ReDimNet 和 Transformers 的相关实现，许可文本见 [inference/licenses](inference/licenses)。模型与数据的使用遵循各自的许可条款。
