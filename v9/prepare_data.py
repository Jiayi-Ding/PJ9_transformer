#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
古诗数据集下载与预处理脚本
- 从 Hugging Face 下载真实唐宋古诗数据
- 构建字符级词表并保存 vocab.json
- 划分 9:1 训练/验证集，编码后保存为 train_data_*.pt / val_data_*.pt
- 改进：为每首诗提取主题标签（边塞/山水/离别/其他），保存主题映射文件

【v5 主题 embedding 改动 1/4】
修改位置：
1. 新增 TOPIC_KEYWORDS 主题关键词库
2. 新增 classify_topic() 主题分类函数
3. 新增 save_topic_mapping() 保存主题标签
4. main() 中调用主题分类并保存 topics_train_{genre}.pt / topics_val_{genre}.pt
5. 新增 topic_vocab.json 保存主题词表

优点：每首诗一个主题标签，训练时整首诗共享同一主题 embedding
"""
"""
【v7】
1.清洗了数据，“无法识别/缺字→小方框代替”这种情况的数据被剔除
"""

"""
【v9】
1. 删除 ci 体裁，只保留五言七言
2. 从本地宋词目录加载六个词牌名的宋词数据，与五言七言统一处理
3. 另外对train和predict做了处理，可以支持手动指定新加词牌名主题
"""

import os
import re
from collections import Counter

# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
import torch
from datasets import load_dataset

try:
    from opencc import OpenCC
except ImportError:
    print("请先安装 opencc：pip install opencc-python-reimplemented")
    sys.exit(1)

cc = OpenCC('t2s')  # 繁体到简体

GENRES = ("5", "7")  # 五言七言体裁

# 方案 2：按类别目标数量补齐
TARGET_PER_GENRE = {"5": 100000, "7": 100000}
MAX_PASS = 5
PASS_SAMPLE_SIZE = 100000

# ========== 目标词牌名列表 ==========
TARGET_CIPAI = ["浣溪沙", "水调歌头", "鹧鸪天", "菩萨蛮", "临江仙", "满江红"]

# ========== 主题关键词库 ==========
TOPIC_KEYWORDS = {
    "landscape": [  # 山水田园
        "山", "水", "江", "河", "湖", "海", "峰", "岭", "岳", "川", "泉", "溪", "涧", "潭",
        "林", "木", "竹", "松", "柏", "梅", "兰", "菊", "荷", "莲", "桃", "柳", "花", "草",
        "苔", "石", "岩", "洞", "野", "原", "田", "园", "村", "庄", "墟", "庐", "茅", "檐",
        "径", "路", "桥", "舟", "船", "渔", "钓", "樵", "耕", "牧", "笛", "琴", "酒", "茶",
        "云", "霞", "烟", "岚", "雨", "雪", "风", "月", "日", "星", "天", "地", "空", "幽",
        "静", "闲", "清", "净", "翠", "碧", "苍", "白"
    ],
    "frontier": [  # 边塞征戍
        "边", "塞", "关", "隘", "城", "戍", "征", "战", "军", "兵", "骑", "马", "弓", "箭",
        "刀", "剑", "戈", "戟", "甲", "铠", "旗", "鼓", "角", "烽", "燧", "火", "烟", "尘",
        "沙", "漠", "碛", "荒", "野", "孤", "寒", "冰", "雪", "霜", "风", "月", "日", "河",
        "山", "陇", "塞上", "玉门", "阳关", "凉州", "陇西", "雁门", "阴山", "楼兰", "天山",
        "胡", "羌", "戎", "狄", "匈奴", "单于", "虏", "敌", "猎", "射", "死", "骨", "血",
        "魂", "归", "望", "乡", "泪", "愁", "老", "壮", "豪", "烈"
    ],
    "homesickness": [  # 思乡怀人
        "乡", "家", "故", "旧", "归", "返", "回", "客", "旅", "羁", "泊", "游", "行", "征",
        "孤", "独", "单", "寂", "寞", "舟", "帆", "船", "车", "马", "驿", "亭", "桥", "关",
        "山", "水", "月", "夜", "秋", "暮", "夕", "昏", "晓", "晨", "霜", "露", "风", "雨",
        "雪", "雁", "鸿", "鱼", "书", "信", "纸", "笔", "梦", "魂", "泪", "泣", "愁", "恨",
        "悲", "哀", "叹", "忆", "念", "思", "怀", "望", "盼", "待", "期", "老", "病", "衰"
    ],
    "historical": [  # 咏史怀古
        "古", "今", "昔", "往", "旧", "故", "前", "后", "朝", "代", "世", "年", "岁", "时",
        "王", "帝", "皇", "侯", "将", "相", "臣", "霸", "雄", "杰", "士", "人", "民", "众",
        "宫", "殿", "阙", "楼", "台", "阁", "城", "郭", "京", "都", "陵", "墓", "碑", "碣",
        "石", "铜", "铁", "金", "玉", "剑", "鼎", "书", "史", "册", "典", "图", "经", "诗",
        "战", "争", "伐", "败", "胜", "兴", "亡", "衰", "盛", "废", "荒", "残", "破", "空",
        "草", "木", "花", "鸟", "燕", "乌", "雀", "江", "河", "山", "月", "风", "烟", "尘"
    ],
    "love": [  # 爱情婚姻
        "心", "情", "思", "念", "忆", "想", "梦", "魂", "意", "愿", "期", "约", "盟", "誓",
        "爱", "怜", "惜", "疼", "亲", "吻", "拥", "抱", "红", "粉", "香", "艳", "丽", "娇",
        "花", "桃", "李", "杏", "梨", "柳", "杨", "燕", "蝶", "鸳", "鸯", "鸾", "凤", "凰",
        "眉", "眼", "脸", "唇", "齿", "发", "鬓", "妆", "镜", "钗", "簪", "环", "佩", "裙",
        "袖", "衣", "裳", "床", "帐", "枕", "被", "帘", "窗", "灯", "烛", "月", "楼", "阁",
        "泪", "泣", "愁", "恨", "怨", "恼", "妒", "痴", "狂", "醉", "醒", "别", "离", "孤"
    ],
    "objects": [  # 咏物言志
        "梅", "兰", "竹", "菊", "松", "柏", "桂", "莲", "荷", "桃", "杏", "李", "柳", "桐",
        "枫", "芦", "苇", "草", "苔", "石", "玉", "金", "银", "铜", "铁", "珠", "宝", "镜",
        "剑", "琴", "棋", "书", "画", "笔", "墨", "纸", "砚", "灯", "烛", "香", "炉", "扇",
        "舟", "车", "桥", "亭", "楼", "台", "雪", "霜", "冰", "露", "月", "星", "云", "霞",
        "龙", "凤", "麟", "龟", "鹤", "鹰", "马", "牛", "蚕", "蜂", "蝉", "蝶", "燕", "雁",
        "节", "骨", "心", "志", "气", "魂", "魄", "清", "高", "洁", "白", "素", "孤", "傲"
    ],
    "farewell": [  # 送别酬唱
        "别", "离", "辞", "去", "往", "行", "远", "长", "短", "送", "饯", "祖", "赠", "留",
        "饮", "酒", "杯", "盏", "壶", "樽", "酌", "醉", "醒", "歌", "唱", "吟", "诗", "赋",
        "和", "答", "酬", "谢", "舟", "船", "帆", "桨", "车", "马", "骑", "桥", "亭", "岸",
        "堤", "柳", "杨", "花", "草", "春", "秋", "朝", "暮", "晓", "夕", "日", "月", "风",
        "烟", "波", "浪", "水", "江", "河", "山", "关", "路", "途", "程", "泪", "泣", "愁",
        "悲", "叹", "忆", "念", "思", "怀", "望", "盼", "逢", "遇", "期", "约", "再", "重"
    ],
    "time_sorrow": [  # 感时伤逝
        "时", "光", "阴", "岁", "年", "载", "春", "夏", "秋", "冬", "季", "节", "朝", "昼",
        "暮", "夕", "昏", "夜", "晓", "晨", "日", "月", "星", "辰", "露", "霜", "雪", "冰",
        "花", "叶", "枝", "条", "草", "木", "荣", "枯", "盛", "衰", "生", "死", "老", "病",
        "少", "壮", "衰", "朽", "残", "破", "落", "谢", "飞", "去", "来", "归", "逝", "流",
        "速", "促", "急", "骤", "忽", "渐", "已", "尽", "空", "虚", "无", "叹", "惜", "悲",
        "伤", "哀", "愁", "恨", "泣", "泪", "梦", "醒", "觉", "悟"
    ],
    "festival": [  # 节序风物
        "春", "夏", "秋", "冬", "年", "岁", "节", "令", "元", "旦", "元夕", "上元", "元宵",
        "寒食", "清明", "端午", "七夕", "中元", "中秋", "重阳", "冬至", "腊日", "除夕",
        "社日", "花朝", "春分", "夏至", "秋分", "冬至", "新", "故", "旧", "换", "更", "易",
        "桃符", "爆竹", "灯", "烛", "月", "饼", "粽", "蒲", "艾", "茱萸", "菊", "酒", "宴",
        "聚", "团", "圆", "乐", "欢", "喜", "庆", "祝", "祈", "愿", "祭", "拜", "游", "赏",
        "踏青", "登高", "竞渡", "赏月", "守岁", "拜年", "送灶", "迎神"
    ],
    "palace_grievance": [  # 宫怨闺怨
        "宫", "殿", "阙", "楼", "台", "阁", "苑", "园", "墙", "柳", "花", "草", "鸟", "燕",
        "莺", "蝶", "日", "月", "星", "云", "风", "雨", "雪", "霜", "春", "夏", "秋", "冬",
        "朝", "暮", "昼", "夜", "晓", "夕", "眠", "睡", "醒", "梦", "妆", "镜", "钗", "簪",
        "裙", "袖", "衣", "裳", "床", "帐", "枕", "被", "帘", "窗", "扉", "户", "灯", "烛",
        "香", "炉", "琴", "瑟", "筝", "笛", "扇", "团扇", "纨扇", "泪", "泣", "泣", "愁",
        "恨", "怨", "悲", "哀", "叹", "独", "孤", "寂", "寞", "冷", "寒", "凉", "老", "衰"
    ],
    "immortal": [  # 游仙求道
        "仙", "神", "真", "灵", "圣", "玄", "道", "丹", "鼎", "药", "炉", "火", "汞", "铅",
        "玉", "石", "珠", "宝", "金", "银", "云", "霞", "霓", "虹", "风", "月", "星", "辰",
        "天", "地", "山", "海", "岛", "洲", "洞", "天", "府", "宫", "殿", "阙", "楼", "台",
        "龙", "凤", "麟", "鹤", "鸾", "龟", "鹿", "芝", "草", "桃", "杏", "花", "露", "泉",
        "酒", "琼浆", "玉液", "瑶池", "蓬莱", "方丈", "瀛洲", "昆仑", "阆苑", "青鸟", "王母",
        "乘", "驾", "飞", "升", "浮", "游", "访", "寻", "采", "炼", "修", "养", "寿", "生",
        "老", "死", "超", "脱", "解", "化", "羽化", "登仙"
    ],
    "zen": [  # 禅理玄思
        "空", "无", "有", "真", "假", "实", "虚", "幻", "梦", "影", "象", "相", "色", "声",
        "香", "味", "触", "法", "心", "性", "命", "理", "道", "德", "仁", "义", "礼", "智",
        "信", "禅", "定", "慧", "戒", "忍", "精", "进", "苦", "集", "灭", "道", "缘", "业",
        "报", "因", "果", "轮", "回", "生", "死", "来", "去", "住", "止", "观", "照", "寂",
        "灭", "静", "净", "清", "明", "光", "灵", "妙", "玄", "幽", "微", "闲", "逸", "淡",
        "寺", "庙", "庵", "院", "僧", "尼", "佛", "菩萨", "罗汉", "塔", "经", "咒", "钵"
    ],
    "drinking": [  # 饮酒狂放
        "酒", "醉", "醒", "饮", "酌", "酹", "觞", "杯", "盏", "壶", "樽", "卮", "觥", "筹",
        "酣", "酩", "酊", "狂", "豪", "放", "逸", "傲", "啸", "笑", "歌", "唱", "吟", "咏",
        "舞", "剑", "琴", "瑟", "鼓", "钟", "月", "花", "雪", "山", "水", "天", "地", "江",
        "河", "海", "云", "风", "春", "秋", "暮", "夕", "白", "老", "少", "壮", "客", "仙",
        "侠", "士", "友", "朋", "独", "孤", "愁", "恨", "悲", "乐", "欢", "忘", "解", "消",
        "浮生", "尘世", "功名", "富贵", "千金", "万古", "百岁", "千杯", "一醉", "何妨"
    ],
    "elegy": [  # 悼亡哭逝
        "悼", "哭", "哀", "悲", "伤", "痛", "绝", "恨", "泣", "涕", "泪", "血", "魂", "魄",
        "灵", "神", "鬼", "幽", "冥", "夜", "梦", "忆", "念", "思", "怀", "寻", "访", "遇",
        "见", "逢", "死", "亡", "殁", "逝", "终", "尽", "绝", "断", "空", "虚", "无", "荒",
        "墓", "坟", "冢", "碑", "碣", "棺", "椁", "骨", "灰", "尘", "土", "草", "木", "露",
        "霜", "雪", "月", "风", "雨", "秋", "暮", "夕", "旧", "故", "昔", "往", "前", "后",
        "君", "卿", "郎", "妾", "妻", "夫", "友", "朋", "师", "父", "母", "兄", "弟", "子"
    ],
    "imperial_exam": [  # 科举入仕
        "科", "举", "试", "考", "策", "论", "诗", "赋", "经", "书", "史", "子", "集", "文",
        "学", "才", "能", "智", "谋", "策", "略", "名", "榜", "题", "名", "登", "第", "及第",
        "进士", "状元", "榜眼", "探花", "秀才", "举人", "贡士", "解元", "会元", "落第", "下第",
        "不第", "罢", "黜", "贬", "谪", "迁", "官", "禄", "位", "职", "权", "势", "名", "利",
        "朝", "廷", "帝", "王", "君", "臣", "宦", "仕", "隐", "退", "归", "田", "书", "剑",
        "灯", "烛", "夜", "晓", "勤", "苦", "寒", "窗", "砚", "墨", "笔", "纸", "卷"
    ],
    "war_atrocity": [  # 战乱纪实
        "战", "乱", "兵", "燹", "火", "焚", "烧", "杀", "戮", "屠", "斩", "伐", "攻", "击",
        "围", "困", "逃", "亡", "流", "离", "失", "所", "散", "奔", "走", "窜", "饥", "饿",
        "殍", "荒", "旱", "涝", "疫", "病", "死", "丧", "骨", "尸", "骸", "血", "泪", "哭",
        "泣", "悲", "哀", "痛", "苦", "愁", "怨", "恨", "民", "众", "苍生", "百姓", "黎民",
        "村", "庄", "城", "郭", "市", "井", "庐", "舍", "屋", "宇", "丘", "墟", "草", "木",
        "野", "兽", "乌", "鸦", "鬼", "魂", "夜", "月", "寒", "风", "霜", "雪"
    ],
    "feminine_life": [  # 女性生活
        "女", "妇", "妾", "婢", "娘", "娇", "媚", "妍", "丽", "艳", "红", "粉", "香", "翠",
        "绿", "青", "黄", "紫", "花", "兰", "梅", "桃", "李", "杏", "柳", "杨", "草", "苔",
        "燕", "莺", "鹊", "蝶", "蜂", "镜", "妆", "梳", "髻", "鬓", "眉", "唇", "脸", "颊",
        "钗", "簪", "环", "珮", "钏", "裙", "衫", "襦", "袄", "裳", "衣", "袖", "帕", "巾",
        "针", "线", "绣", "织", "纺", "绩", "灯", "烛", "窗", "帘", "床", "帐", "枕", "席",
        "琴", "筝", "瑟", "笙", "箫", "棋", "书", "画", "诗", "词", "春", "秋", "月", "夜"
    ],
    "reclusion_tourism": [  # 隐逸游赏
        "隐", "逸", "遁", "藏", "避", "逃", "退", "休", "闲", "静", "寂", "默", "淡", "泊",
        "清", "高", "孤", "傲", "野", "山", "水", "林", "泉", "石", "岩", "洞", "谷", "涧",
        "溪", "湖", "潭", "洲", "渚", "汀", "岛", "寺", "庙", "观", "庵", "院", "僧", "道",
        "鹤", "鹿", "猿", "鸟", "花", "草", "竹", "松", "梅", "兰", "菊", "莲", "云", "霞",
        "烟", "岚", "风", "月", "星", "露", "霜", "雪", "舟", "船", "车", "马", "杖", "屐",
        "酒", "茶", "琴", "棋", "书", "画", "诗", "赋", "游", "赏", "登", "临", "访", "寻"
    ],
}

# 顺序固定，用于 id 映射（训练时 NUM_TOPICS 会从这里读取）
TOPIC_LIST = [
    "landscape",           # 山水田园
    "frontier",            # 边塞征戍
    "homesickness",        # 思乡怀人
    "historical",          # 咏史怀古
    "love",                # 爱情婚姻
    "objects",             # 咏物言志
    "farewell",            # 送别酬唱
    "time_sorrow",         # 感时伤逝
    "festival",            # 节序风物
    "palace_grievance",    # 宫怨闺怨
    "immortal",            # 游仙求道
    "zen",                 # 禅理玄思
    "drinking",            # 饮酒狂放
    "elegy",               # 悼亡哭逝
    "imperial_exam",       # 科举入仕
    "war_atrocity",        # 战乱纪实
    "feminine_life",       # 女性生活
    "reclusion_tourism",   # 隐逸游赏
    "other",               # 其他（未分类或综合类）
]
NUM_TOPICS = len(TOPIC_LIST)


def classify_topic(text: str) -> str:
    """
    根据关键词匹配，判断诗歌主题。
    返回: frontier / landscape / farewell /……/ other
    """
    # 统计各主题关键词出现次数
    scores = {topic: 0 for topic in TOPIC_LIST[:-1]}  # other 不参与计数
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            # 简单匹配：关键词出现在文本中
            if kw in text:
                scores[topic] += 1
    
    # 找出得分最高的主题（且得分>0），否则返回 other
    max_topic = max(scores, key=scores.get)
    if scores[max_topic] > 0:
        return max_topic
    return "other"


def normalize_poem_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("\r", "")
    # 繁体转简体
    text = cc.convert(text)
    return text


def is_valid_poem(text: str) -> bool:
    """
    判断一首诗/词是否有效（无缺字、无全缺字行、长度合理）
    """
    if not text or len(text.strip()) == 0:
        return False

    # 1. 包含方框缺字标记 → 无效
    if '□' in text:
        return False

    lines = text.split('\n')
    total_chinese = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 统计该行中的中文字符数
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', line))
        total_chinese += chinese_count
        # 2. 该行无中文字符 → 无效（全缺字行或纯标点行）
        if chinese_count == 0:
            return False

    # 3. 整首诗的中文字符过少 → 无效
    if total_chinese < 10:
        return False

    return True


def split_poem_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return lines
    # 若无换行，则按常见标点分句
    segments = re.split(r"[，。；？！、]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def classify_poem(text: str) -> str:
    """
    对一首诗进行分类：5 言、7 言或词/杂言。
    规则：按句长比例判断，若 5 言句占比 >= 0.6 则为 5 言，7 言句占比 >= 0.6 则为 7 言，否则为词。
    """
    text = normalize_poem_text(text)
    lines = split_poem_lines(text)
    if not lines:
        return "ci"

    counts = [count_chinese_chars(line) for line in lines]
    counts = [c for c in counts if c > 0]
    if not counts:
        return "ci"

    n5 = sum(1 for c in counts if c == 5)
    n7 = sum(1 for c in counts if c == 7)
    total = len(counts)

    if n5 / total >= 0.6:
        return "5"
    if n7 / total >= 0.6:
        return "7"
    return "ci"


def write_text_file(path: str, poems: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(poems))


def encode_and_save(text: str, stoi: dict, out_path: str) -> None:
    ids = [stoi[c] for c in text]
    data = torch.tensor(ids, dtype=torch.long)
    torch.save(data, out_path)


def load_cipai_from_local() -> dict:
    """
    从当前目录下的 '宋词' 文件夹（chinese-poetry 项目结构）中加载六个词牌名的宋词数据。
    假设目录结构: ./宋词/*.json
    返回: {词牌名: [词文本列表]}
    """
    print("\n" + "=" * 50)
    print("步骤 2.5：从本地 '宋词' 目录加载宋词数据（六个词牌名）")
    print("=" * 50)
    
    ci_dir = os.path.join(os.path.dirname(__file__), "宋词")
    if not os.path.isdir(ci_dir):
        print(f"警告: 找不到宋词目录: {ci_dir}，将跳过词牌数据加载。", file=sys.stderr)
        return {}

    import glob
    cipai_dict = {cp: [] for cp in TARGET_CIPAI}
    total_valid = 0

    json_files = glob.glob(os.path.join(ci_dir, "*.json"))
    if not json_files:
        print(f"警告: 在 {ci_dir} 下未找到任何 JSON 文件。", file=sys.stderr)
        return {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 数据可能是列表形式
                if isinstance(data, list):
                    for item in data:
                        rhythmic = item.get("rhythmic", "").strip()
                        rhythmic = cc.convert(rhythmic)   # 繁转简
                        if rhythmic not in TARGET_CIPAI:
                            continue
                        paragraphs = item.get("paragraphs", [])
                        if not paragraphs:
                            continue
                        # 用换行符连接段落
                        text = "".join(paragraphs)
                        text = normalize_poem_text(text)
                        if not is_valid_poem(text):
                            continue
                        cipai_dict[rhythmic].append(text)
                        total_valid += 1
        except Exception as e:
            print(f"警告: 处理文件 {json_file} 时出错: {e}", file=sys.stderr)

    print(f"成功加载有效宋词: {total_valid} 首")
    for cp, poems in cipai_dict.items():
        print(f"  {cp}: {len(poems)} 首")
    return cipai_dict


def main():
    print("=" * 50)
    print("步骤 1：下载真实唐宋古诗数据集（五言七言）")
    print("=" * 50)
    dataset = load_dataset("Lifan-Z/Chinese-poetries-txt", split="train")
    print(f"数据集总条数: {len(dataset)}")
    print(f"列名: {dataset.column_names}")

    dataset = dataset.shuffle(seed=42)
    text_col = "text" if "text" in dataset.column_names else dataset.column_names[0]
    print(f"使用列: {text_col}")

    print("\n" + "=" * 50)
    print("步骤 2：按五言七言体裁目标数量补齐样本")
    print("=" * 50)

    poems_by_genre = {g: [] for g in GENRES}
    all_texts = []
    seen_texts = set()
    targets = TARGET_PER_GENRE.copy()
    print(f"目标每类样本数: {targets}")

    added = True
    filtered_count = 0
    for pass_i in range(MAX_PASS):
        if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
            break
        if not added and pass_i > 0:
            break
        added = False

        print(f"Pass {pass_i + 1}/{MAX_PASS}：随机抽样最多 {PASS_SAMPLE_SIZE} 条，按体裁补齐")
        shuffled = dataset.shuffle(seed=42 + pass_i)
        if len(shuffled) > PASS_SAMPLE_SIZE:
            shuffled = shuffled.select(range(PASS_SAMPLE_SIZE))

        for item in shuffled:
            text = item[text_col]
            if text is None:
                continue
            text = normalize_poem_text(text)
            if not text or text in seen_texts:
                continue
            if not is_valid_poem(text):
                filtered_count += 1
                continue
            seen_texts.add(text)
            genre = classify_poem(text)
            if genre not in GENRES:
                continue
            if len(poems_by_genre[genre]) < targets[genre]:
                poems_by_genre[genre].append(text)
                all_texts.append(text)
                added = True
            if all(len(poems_by_genre[g]) >= targets[g] for g in GENRES):
                break

    print(f"过滤掉的无效诗歌数: {filtered_count}")
    print(f"实际抽样五言七言条数: {len(all_texts)}")

    # ========== 加载词牌数据 ==========
    cipai_data = load_cipai_from_local()
    for cp, poems in cipai_data.items():
        if poems:
            poems_by_genre[cp] = poems
            all_texts.extend(poems)

    # 汇总
    total_poems = sum(len(poems) for poems in poems_by_genre.values())
    print(f"\n总计有效作品数（五言+七言+词牌）: {total_poems}")

    all_text = "\n\n".join(all_texts)
    with open("poetry.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"已保存 poetry.txt，总字符数约: {len(all_text)}")
    for genre, poems in poems_by_genre.items():
        print(f"  {genre}: {len(poems)} 首")

    print("\n" + "=" * 50)
    print("步骤 3：构建字符级词表并保存 vocab.json（基于所有文本）")
    print("=" * 50)
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    vocab = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词表大小: {vocab_size}")

    # 保存主题词表
    print("\n" + "=" * 50)
    print("步骤 3.5：保存主题词表 topic_vocab.json")
    print("=" * 50)
    topic_vocab = {
        "topics": TOPIC_LIST,
        "num_topics": NUM_TOPICS,
        "topic_keywords": TOPIC_KEYWORDS  # 保存关键词供参考
    }
    with open("topic_vocab.json", "w", encoding="utf-8") as f:
        json.dump(topic_vocab, f, ensure_ascii=False, indent=2)
    print(f"已保存主题词表: topic_vocab.json (主题数: {NUM_TOPICS})")

    print("\n" + "=" * 50)
    print("步骤 4：按体裁（五言、七言、各词牌名）划分 9:1 训练/验证集并保存 .pt 文件")
    print("=" * 50)

    for genre, poems in poems_by_genre.items():
        print(f"\n处理体裁/词牌: {genre}，共 {len(poems)} 首")
        if not poems:
            print(f"  跳过 {genre}，无数据。")
            continue

        # 保存纯文本
        write_text_file(f"poetry_{genre}.txt", poems)

        # 划分训练/验证集
        n_train = int(len(poems) * 0.9)
        train_poems = poems[:n_train]
        val_poems = poems[n_train:]

        train_text = "\n\n".join(train_poems)
        val_text = "\n\n".join(val_poems)

        # 构建边界
        train_boundaries = []
        pos = 0
        for poem in train_poems:
            train_boundaries.append(pos)
            pos += len(poem) + 2   # "\n\n"

        val_boundaries = []
        pos = 0
        for poem in val_poems:
            val_boundaries.append(pos)
            pos += len(poem) + 2

        # 编码并保存字符序列
        encode_and_save(train_text, stoi, f"train_data_{genre}.pt")
        encode_and_save(val_text, stoi, f"val_data_{genre}.pt")

        print(f"  保存: train_data_{genre}.pt, val_data_{genre}.pt")
        print(f"  训练集: {len(train_poems)} 首, token 数: {len(train_text)}")
        print(f"  验证集: {len(val_poems)} 首, token 数: {len(val_text)}")
        print(f"训练边界数: {len(train_boundaries)}")
        print(f"验证边界数: {len(val_boundaries)}")

        # 主题标签
        train_topics = []
        for poem in train_poems:
            topic = classify_topic(poem)
            topic_id = TOPIC_LIST.index(topic)
            train_topics.append(topic_id)

        val_topics = []
        for poem in val_poems:
            topic = classify_topic(poem)
            topic_id = TOPIC_LIST.index(topic)
            val_topics.append(topic_id)

        train_topics_tensor = torch.tensor(train_topics, dtype=torch.long)
        val_topics_tensor = torch.tensor(val_topics, dtype=torch.long)
        train_boundaries = torch.tensor(train_boundaries, dtype=torch.long)
        val_boundaries = torch.tensor(val_boundaries, dtype=torch.long)

        torch.save(train_topics_tensor, f"topics_train_{genre}.pt")
        torch.save(val_topics_tensor, f"topics_val_{genre}.pt")
        torch.save(train_boundaries, f"boundaries_train_{genre}.pt")
        torch.save(val_boundaries, f"boundaries_val_{genre}.pt")

        # 打印主题分布统计
        train_topic_names = [TOPIC_LIST[tid] for tid in train_topics]
        train_dist = Counter(train_topic_names)
        val_topic_names = [TOPIC_LIST[tid] for tid in val_topics]
        val_dist = Counter(val_topic_names)
        print(f"  训练集主题分布: {dict(train_dist)}")
        print(f"  验证集主题分布: {dict(val_dist)}")

    print("\n预处理全部完成。")
    print("\n提示：已生成主题标签文件，训练时请使用 --use_topic 参数启用主题 embedding")


if __name__ == "__main__":
    main()