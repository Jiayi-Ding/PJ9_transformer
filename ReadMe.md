# ReadMe

## 环境安装

pip install torch datasets opencc-python-reimplemented pypinyin pingshui_rhyme

## 运行步骤（按顺序）

1. 准备数据
python prepare_data.py

2. 构建韵母映射（押韵功能需要）
python build_rhyme_dict.py

3. 构建平仄映射（平仄约束需要）
python build_pingze_dict.py

4. 训练模型
python train.py --genre 5                  # 五言
python train.py --genre 7 --use_topic      # 七言（启用主题）

## 预测指令

基础生成：
python predict.py --genre 5 --prompt 春 --lines 4
python predict.py --genre 7 --prompt 月 --lines 8

指定主题：
python predict.py --genre 5 --prompt 春 --topic landscape

自动选择主题（根据起笔字）：
python predict.py --genre 5 --prompt 春

押韵：
python predict.py --genre 5 --prompt 春 --rhyme an          # 指定韵脚
python predict.py --genre 5 --prompt 春 --auto_rhyme        # 自动押韵
python predict.py --genre 5 --prompt 春 --no_rhyme          # 禁用押韵

平仄约束：
python predict.py --genre 5 --prompt 春 --use_pingze --lines 4
python predict.py --genre 5 --prompt 春 --use_pingze --auto_rhyme --lines 4

完整示例：
python predict.py --genre 5 --prompt 春 --topic landscape --auto_rhyme --use_pingze --lines 4 --temperature 0.8 --rep_penalty 1.2

## 参数说明

体裁：--genre 5（五言）或 7（七言）
起笔：--prompt 任意汉字
主题：--topic landscape / frontier / homesickness / historical / love / objects / farewell / time_sorrow / festival / palace_grievance / immortal / zen / drinking / elegy / imperial_exam / war_atrocity / feminine_life / reclusion_tourism / other
行数：--lines 4（绝句）或 8（律诗）
押韵：--rhyme an/ang/ong 指定韵脚，--auto_rhyme 自动押韵，--no_rhyme 禁用
平仄：--use_pingze 启用
温度：--temperature 0.8（默认）（这是base-temperature）
重复惩罚：--rep_penalty 1.2（默认）
长度：--max_new 200（默认）