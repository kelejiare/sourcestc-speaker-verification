# 推理代码

`run.py` 负责连接声学模型、适配器和教师特征模块；`batch_submission_runtime.py` 负责分组校正与结果写入。

从仓库根目录执行：

```bash
XFDATA_ROOT=/path/to/sourcestc-test bash scripts/infer.sh
```

模型说明见 [技术方案](../docs/METHOD.md)，环境与输入格式见 [训练与运行说明](../docs/REPRODUCIBILITY.md)。
