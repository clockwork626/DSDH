# -*- coding: utf-8 -*-
"""
任务二：比较李白与杜甫的文本特征（TF-IDF）
第一步：从 chinese-poetry 库提取两人的诗，清洗后各随机抽 100 首，存 libai_dufu_poems.csv
第二步：jieba 分词 + 停用词过滤 + TfidfVectorizer，取各自 Top10 高权重词，画对比条形图

依赖（缺什么装什么）：
    pip install pandas jieba scikit-learn matplotlib opencc-python-reimplemented

用法（在 DSDH 根目录下运行）：
    python libai_dufu_tfidf.py            # 完整流程（CSV 不存在时先执行第一步）
    python libai_dufu_tfidf.py step1      # 只执行数据提取与清洗
    python libai_dufu_tfidf.py step2      # 只执行 TF-IDF 分析与可视化
"""
import csv
import glob
import json
import os
import random
import re
import sys

# ---------------- 配置 ----------------
AUTHORS = ["李白", "杜甫"]
SAMPLE_N = 100          # 每位作者抽取的篇数
RANDOM_STATE = 42       # 随机种子，保证可复现
CSV_FILE = "libai_dufu_poems.csv"
PNG_FILE = "libai_dufu_tfidf.png"

# 古文停用词表：覆盖常见文言虚词、代词、介词、连词与泛义动词
STOPWORDS = set("""
之 其 以 于 而 则 乃 若 且 矣 乎 哉 焉 耳 吾 汝 尔 我 余 予 卿 君 谁
在 得 见 又 亦 向 皆 尽 道 坐 垂 相 随 未 无 不 有 何 此 所 为
与 也 者 夫 彼 自 或 己 已 还 将 更 中 间 上 下 里 来 去 一 人 山 水
惟 唯 盖 遂 犹 尚 辄 即 固 必 俱 悉 便 始 方 正 曾
何处 不见 不知 不得 不可 无人 不及
""".split())

# 异体字归一化表：opencc 只做繁→简转换，不处理异体字（如 疎→疏、劒→剑）
# 依据语料实际扫描结果整理
VARIANT_MAP = {
    "疎": "疏", "劒": "剑", "飜": "翻", "懽": "欢", "驩": "欢",
    "幷": "并", "竝": "并", "徧": "遍", "遶": "绕", "迳": "径", "逈": "迥",
    "隣": "邻", "呌": "叫", "澁": "涩", "簷": "檐", "筯": "箸", "椀": "碗",
    "醆": "盏", "飡": "餐", "餙": "饰", "喫": "吃", "噐": "器", "卽": "即",
    "顦": "憔", "顇": "悴", "媿": "愧", "棃": "梨", "畧": "略", "宂": "冗",
    "臯": "皋", "蘂": "蕊", "緜": "绵", "輭": "软", "韤": "袜", "鼇": "鳌",
    "埽": "扫", "墖": "塔", "勅": "敕", "歛": "敛", "暎": "映", "讐": "仇",
    "谿": "溪", "遯": "遁", "𬬻": "炉",
}
VARIANT_TABLE = str.maketrans(VARIANT_MAP)


# ---------------- 第一步：数据提取与清洗 ----------------
def find_tang_files():
    """定位全唐诗 JSON 文件。

    只取「全唐诗/poet.tang.*.json」；
    注意：strains/json/ 下有同名文件，那是平仄声调数据，不是诗文，必须排除。
    兼容 chinese-poetry 与 chinese-poetry-master 两种文件夹命名及嵌套。
    """
    bases = {os.getcwd(), os.path.dirname(os.path.abspath(__file__))}
    patterns = []
    for base in bases:
        for root in ("chinese-poetry", "chinese-poetry-master"):
            patterns.append(os.path.join(base, root, "全唐诗", "poet.tang.*.json"))
            patterns.append(os.path.join(base, root, root, "全唐诗", "poet.tang.*.json"))
    files = sorted({os.path.normcase(os.path.normpath(f))
                    for p in patterns for f in glob.glob(p)})
    return files


def clean_text(text):
    """去除标点、空格等一切非汉字字符，只保留汉字（正则处理）。"""
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def get_t2s():
    """繁体转简体。全唐诗语料是繁体，转换后 jieba 分词与简体停用词表才有效。
    opencc 缺失时回退为原样返回，并给出提示。"""
    try:
        from opencc import OpenCC
        cc = OpenCC("t2s")
        return cc.convert
    except ImportError:
        print("提示：未安装 opencc-python-reimplemented，文本将保持繁体，"
              "分词效果会打折扣（建议 pip install opencc-python-reimplemented）。")
        return lambda s: s


def step1_extract():
    files = find_tang_files()
    if not files:
        sys.exit("未找到 全唐诗/poet.tang.*.json，请确认 chinese-poetry 文件夹与本脚本的相对位置。")
    print(f"发现全唐诗文件 {len(files)} 个，开始筛选 {AUTHORS} ...")

    poems = {a: [] for a in AUTHORS}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            a = rec.get("author")
            if a in poems:
                text = "".join(rec.get("paragraphs", []))  # 合并 paragraphs 字段
                poems[a].append(text)

    counts = {a: len(poems[a]) for a in AUTHORS}
    print("筛选结果：", counts)
    for a in AUTHORS:
        if counts[a] < SAMPLE_N:
            sys.exit(f"{a} 存诗 {counts[a]} 首，不足 {SAMPLE_N} 首。")

    to_simplified = get_t2s()
    rng = random.Random(RANDOM_STATE)  # 固定种子，保证可复现

    rows = []
    for a in AUTHORS:
        picked = rng.sample(poems[a], SAMPLE_N)
        # 清洗 -> 繁转简 -> 异体字归一化
        rows.extend((a, to_simplified(clean_text(t)).translate(VARIANT_TABLE))
                    for t in picked)

    with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["author", "text"])
        w.writerows(rows)
    print(f"已写入 {CSV_FILE}：每位作者各 {SAMPLE_N} 首（random_state={RANDOM_STATE}）。")


# ---------------- 第二步：TF-IDF 分析与可视化 ----------------
def step2_analyze():
    import numpy as np
    import pandas as pd
    import jieba
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    print(f"读入 {CSV_FILE}：{len(df)} 首，分布：{df['author'].value_counts().to_dict()}")

    def tokenize(text):
        # 统一分词与停用词规则：jieba 分词 -> 只留纯汉字词 -> 过滤停用词
        return [t for t in jieba.lcut(str(text))
                if re.fullmatch(r"[\u4e00-\u9fff]+", t)
                and t not in STOPWORDS]

    df["tokens"] = df["text"].apply(tokenize)
    corpus = df["tokens"].apply(" ".join)

    # 只保留双字及以上的词（过滤全部单字），停用词表直接传给 TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",
                          stop_words=list(STOPWORDS))
    X = vec.fit_transform(corpus)
    vocab = np.array(vec.get_feature_names_out())
    print(f"词表规模：{len(vocab)} 个词。")

    results = {}
    for a in AUTHORS:
        mask = np.asarray(df["author"] == a)  # 转 numpy 布尔数组才能索引稀疏矩阵
        mean_w = np.asarray(X[mask].mean(axis=0)).ravel()
        top = np.argsort(mean_w)[::-1][:10]
        results[a] = (vocab[top], mean_w[top])
        print(f"\n{a} TF-IDF 高权重词 Top10：")
        for w_, v in zip(vocab[top], mean_w[top]):
            print(f"    {w_:<6}{v:.4f}")

    # 中文字体 + 双面板水平条形图
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, a in zip(axes, AUTHORS):
        words, weights = results[a]
        ax.barh(words, weights, color="#4C72B0" if a == "李白" else "#C44E52")
        ax.invert_yaxis()  # 权重最高的显示在最上方
        ax.set_title(f"{a} TF-IDF 高权重词 Top10")
        ax.set_xlabel("平均 TF-IDF 权重")
        ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.savefig(PNG_FILE, dpi=150)
    print(f"\n图表已保存：{PNG_FILE}")
    plt.show()


# ---------------- 入口 ----------------
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("step1", "all") and not os.path.exists(CSV_FILE):
        step1_extract()
    elif arg == "step1":
        step1_extract()
    if arg in ("step2", "all"):
        if not os.path.exists(CSV_FILE):
            sys.exit(f"缺少 {CSV_FILE}，请先执行 step1。")
        step2_analyze()
