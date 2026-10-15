# 训练代码索引

## 多域 KD、联合与来源感知适配器
- `source/tools/train_redim_output_kd_multidomain.py`
- `source/tools/export_joint_release_adapter.py`
- `source/tools/export_mobile_adapter.py`

## 短时长 CML/PT 适配器
- `source/tools/hierarchical_cml_search.py`
- `source/tools/export_release_adapter.py`
- `source/tools/export_adapter.py`

## ReDimNet2Plus 特征缓存与融合校验
- `source/tools/build_plus_cache.py`
- `source/tools/evaluate_redimnet2plus_fusion.py`

## w2v-BERT 2.0 教师缓存与 LoRA
- `source/tools/prepare_w2vbert2_teacher.py`
- `source/tools/build_teacher_cache.py`
- `source/tools/prepare_data.py`
- `source/tools/train_lora.py`
- `source/tools/evaluate.py`

## 来源/时长策略冻结
- `source/tools/alpha_sweep.py`
- `source/tools/policy_sweep.py`

## 教师子空间均值校正校验
- `source/tools/evaluate_transductive_centering.py`

主干与融合策略见 [技术方案](../docs/METHOD.md)，数据和缓存的准备方式见 [训练与运行说明](../docs/REPRODUCIBILITY.md)。入口所用的辅助代码位于 `source/`。
