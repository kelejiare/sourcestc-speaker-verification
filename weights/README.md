# 模型权重

模型包：`sourcestc-weights.tar.gz`，约 1.51 GB，包含 11 个声学主干、教师模型和适配器权重文件。

[百度网盘下载](https://pan.baidu.com/s/1NMlsouDgTRTS9uIV47wl1w?pwd=v464)，提取码：`v464`。

将压缩包放入本目录。运行 `bash scripts/infer.sh` 时会自动准备权重，生成 `weights/prepared/model/`。

权重位于其他位置时，可通过环境变量指定：

```bash
MODEL_PACKAGE=/path/to/sourcestc-weights.tar.gz \
XFDATA_ROOT=/path/to/sourcestc-test \
bash scripts/infer.sh
```
