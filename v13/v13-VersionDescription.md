# v13-平仄：版本说明（2026年6月5日）

---

## 1. 基于 v12 作出更改

在押韵功能的基础上，增加了**平仄约束**，使生成的古诗能够符合近体诗的平仄格律（如“仄仄平平仄，平平仄仄平”）。  
目前仅支持五言/七言的绝句（4句）和律诗（8句），暂不支持词牌。  
平仄约束与押韵约束可以**同时启用**，若冲突则优先保证**押韵**。

---

## 2. 改动原理

平仄约束采用**纯解码约束**方案（无需重新训练模型），通过预置的标准平仄模板，在生成时强制每个位置的汉字必须符合期望的平仄（0=平，1=仄），标点位置自动跳过约束。

### 核心改动点

1. **构建平仄映射表**  
   新建 `build_pingze_dict.py`，基于 `pingshui_rhyme` 库（平水韵）为每个汉字标注平仄，若失败则依次降级为：繁体转换 → 拼音法（普通话声调）。最终生成 `pingze_dict.json`。

2. **`PingzeHelper` 辅助类**  
   加载平仄字典，提供 `get_tokens_by_pingze(expected)` 方法，返回所有平仄为 `expected` 的 token id 列表（预先缓存，性能优化）。

3. **平仄模板定义**  
   在 `predict.py` 中预定义了五言/七言的**绝句**和**律诗**的标准平仄格式（含平起式和仄起式），并根据起笔字的平仄自动选择合适的首句格式。

4. **生成时动态计算期望平仄**  
   维护 `chinese_count`（已生成的中文字符数），根据 `line_length`（5或7）计算出当前位于第几句、句内第几个字，查表得到期望平仄。  
   关键修复：当 `chinese_count % line_length == 0` 且 `chinese_count > 0` 时，下一个字符应为标点，不施加平仄约束。

5. **平仄与押韵协同**  
   - 若当前位置**同时有平仄约束和押韵需求**，则取两者允许 token 集合的**交集**；若交集为空，则**放弃押韵，保留平仄**，并打印警告（仅一次）。  
   - 自动押韵时，韵脚字的采样也会受平仄约束（只从符合平仄的汉字中采样）。

6. **通用约束采样函数**  
   将原有的 `sample_rhyme_only` 泛化为 `sample_with_allowed`，同时服务于平仄和押韵，支持温度缩放、重复惩罚、跳过特殊 token。

---

## 3. 新增/修改的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `build_pingze_dict.py` | **新建** | 基于平水韵库构建汉字→平仄映射表（0平1仄2其他） |
| `predict.py` | **修改** | 增加平仄模板、`PingzeHelper` 类、约束采样函数，修改 `generate_one` 集成平仄逻辑 |
| `pingze_dict.json` | **生成** | 由 `build_pingze_dict.py` 输出，供预测时加载 |

**未修改**：`prepare_data.py`, `dataset.py`, `model.py`, `train.py`, `rhyme_utils.py`

---

## 4. 注意事项

- 需要安装额外依赖：`pingshui_rhyme` 和 `pypinyin`（若拼音法兜底）。  
- 必须先运行 `prepare_data.py` 生成 `vocab.json`，再运行 `build_pingze_dict.py` 生成平仄字典。  
- 平仄约束**仅对五言/七言有效**，词牌或杂言体裁使用 `--genre ci` 时不会启用。  
- 律诗（8句）的平仄模板基于“仄起式”预定义，平起式暂时复用仄起式（实际效果影响不大）。  
- 若 `--use_pingze` 与 `--auto_rhyme` 或 `--rhyme` 同时使用，会尽量同时满足，冲突时以平仄优先。

---

## 5. 命令行示例

```bash
# 仅平仄（五言绝句，根据起笔字“春”自动选择平起或仄起）
python predict.py --genre 5 --prompt 春 --use_pingze --lines 4

# 平仄 + 自动押韵（七言律诗）
python predict.py --genre 7 --prompt 月 --auto_rhyme --use_pingze --lines 8

# 平仄 + 指定韵脚 ang（五言绝句）
python predict.py --genre 5 --prompt 雪 --rhyme ang --use_pingze --lines 4

# 仅押韵，不平仄
python predict.py --genre 5 --prompt 春 --auto_rhyme --lines 4

# 完整参数示例（主题+平仄+押韵+重复惩罚）
python predict.py --genre 7 --prompt 江 --topic landscape --auto_rhyme --use_pingze --rep_penalty 1.3 --lines 8
```

---

## 6. 完整流程示例

```bash
# 1. 安装依赖（若尚未安装 pingshui_rhyme）
pip install torch datasets opencc-python-reimplemented pypinyin matplotlib pingshui_rhyme

# 2. 数据准备（约2-5分钟）
python prepare_data.py

# 3. 构建韵母表（押韵需要，约30秒）
python build_rhyme_dict.py

# 4. 构建平仄映射表（新增步骤，约30秒）
python build_pingze_dict.py

# 5. 训练五言模型（约3-5分钟）
python train.py --genre 5 --use_topic --epochs 3 --batch_size 32 --train_num_samples 50000

# 6. 训练七言模型（约3-5分钟）
python train.py --genre 7 --use_topic --epochs 3 --batch_size 32 --train_num_samples 50000

# 7. 测试生成（平仄+押韵）
python predict.py --genre 5 --prompt 春 --topic landscape --auto_rhyme --use_pingze --lines 4
python predict.py --genre 7 --prompt 月 --topic frontier --rhyme ang --use_pingze --lines 8
```

---