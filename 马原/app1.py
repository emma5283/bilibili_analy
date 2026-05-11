import streamlit as st
import pandas as pd
import numpy as np
import jieba
from pathlib import Path

from snownlp import SnowNLP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

import plotly.express as px
import matplotlib.pyplot as plt

# =====================================
# 页面设置
# =====================================

st.set_page_config(
    page_title="历史传播分析教材",
    layout="wide"
)

# =====================================
# 教材风格CSS
# =====================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #f6f1e7;
    font-family: "STSong", "SimSun", serif;
}

.main {
    background-color: #f6f1e7;
}

h1 {
    color: #2f2f2f;
    font-size: 52px !important;
    font-weight: bold;
    margin-bottom: 20px;
}

h2 {
    color: #b5532d;
    border-left: 8px solid #b5532d;
    padding-left: 15px;
    margin-top: 40px;
}

h3 {
    color: #8b4513;
}

[data-testid="stMetric"] {
    background-color: #fffdf8;
    border: 2px solid #d6c7a1;
    padding: 15px;
    border-radius: 12px;
}

.stDataFrame {
    background-color: white;
}

.conclusion-box {
    background-color: #fff8e8;
    padding: 25px;
    border-left: 10px solid #b5532d;
    margin-top: 20px;
    margin-bottom: 20px;
    font-size: 20px;
    line-height: 2;
    color: #3a3127;
    font-family: "STSong", "SimSun", serif;
}

.book-box {
    background-color: #fff8e8;
    border: 2px solid #d8c8aa;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    font-family: "STSong", "SimSun", serif;
    color: #3a3127;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# 标题
# =====================================

st.markdown("""
<h1>
第1课<br>
平台时代的历史叙事变迁
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div class="book-box">
本系统通过对B站历史区高播放视频进行分析，研究：

- 流量机制如何改变历史叙事
- 什么样的历史更容易传播
- 平台如何塑造公众历史认知
- 情绪化传播如何压缩历史复杂性
""", unsafe_allow_html=True)

# =====================================
# 读取数据
# =====================================

file_path = Path(__file__).with_name("videoinfo(2).xlsx")

df = pd.read_excel(file_path)

# 清理列名

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "")
)

# =====================================
# 数值化
# =====================================

numeric_cols = [
    "播放数",
    "点赞数",
    "收藏",
    "转发",
    "弹幕数",
    "投硬币",
    "时长(秒)"
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# =====================================
# 互动率
# =====================================

df["互动率"] = (
    df["点赞数"] +
    df["收藏"] +
    df["转发"] +
    df["投硬币"]
) / df["播放数"]

if "时长(秒)" in df.columns:

    df["时长(分钟)"] = df["时长(秒)"] / 60

    duration_bins = [0, 60, 180, 300, 600, 1200, np.inf]
    duration_labels = [
        "1分钟以内",
        "1-3分钟",
        "3-5分钟",
        "5-10分钟",
        "10-20分钟",
        "20分钟以上"
    ]

    df["时长类型"] = pd.cut(
        df["时长(秒)"],
        bins=duration_bins,
        labels=duration_labels,
        right=False
    )

# =====================================
# 自动情绪分析
# =====================================


def get_sentiment(text):

    try:
        s = SnowNLP(str(text))
        return s.sentiments

    except:
        return np.nan


df["自动情绪得分"] = df["标题"].apply(get_sentiment)

# =====================================
# 首页数据
# =====================================

st.header("学习聚焦")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "历史视频数量",
    len(df)
)

col2.metric(
    "平均播放量",
    f"{df['播放数'].mean():,.0f}"
)

col3.metric(
    "平均互动率",
    f"{df['互动率'].mean():.2%}"
)

col4.metric(
    "UP主数量",
    df["UP主"].nunique()
)

# =====================================
# 动态互动率图
# =====================================

st.header("历史视频互动率结构")

fig1 = px.scatter(
    df,
    x="播放数",
    y="互动率",
    color="题材" if "题材" in df.columns else None,
    size="点赞数",
    hover_data=["标题", "UP主"],
    title="播放量与互动率关系"
)

fig1.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =====================================
# 时长分析
# =====================================

if "时长(秒)" in df.columns:

    st.header("历史视频时长结构分析")

    duration_df = df.dropna(subset=["时长(秒)", "时长(分钟)"]).copy()

    duration_col1, duration_col2, duration_col3, duration_col4 = st.columns(4)

    duration_col1.metric(
        "平均时长",
        f"{duration_df['时长(分钟)'].mean():.1f} 分钟"
    )

    duration_col2.metric(
        "中位时长",
        f"{duration_df['时长(分钟)'].median():.1f} 分钟"
    )

    duration_col3.metric(
        "短视频占比",
        f"{(duration_df['时长(秒)'] < 180).mean():.2%}"
    )

    duration_col4.metric(
        "长视频占比",
        f"{(duration_df['时长(秒)'] >= 600).mean():.2%}"
    )

    fig_duration_hist = px.histogram(
        duration_df,
        x="时长(分钟)",
        nbins=30,
        color="时长类型",
        title="历史视频时长分布"
    )

    fig_duration_hist.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig_duration_hist,
        use_container_width=True
    )

    fig_duration_scatter = px.scatter(
        duration_df,
        x="时长(分钟)",
        y="播放数",
        color="题材" if "题材" in duration_df.columns else "时长类型",
        size="互动率",
        hover_data=["标题", "UP主", "互动率"],
        title="视频时长与播放量关系"
    )

    fig_duration_scatter.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig_duration_scatter,
        use_container_width=True
    )

    duration_group = duration_df.groupby(
        "时长类型",
        observed=False
    ).agg({
        "播放数": "mean",
        "互动率": "mean",
        "点赞数": "mean",
        "标题": "count"
    }).reset_index()

    duration_group.columns = [
        "时长类型",
        "平均播放数",
        "平均互动率",
        "平均点赞数",
        "视频数量"
    ]

    fig_duration_bar = px.bar(
        duration_group,
        x="时长类型",
        y="平均播放数",
        color="平均互动率",
        text="视频数量",
        title="不同时长类型的平均传播表现"
    )

    fig_duration_bar.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig_duration_bar,
        use_container_width=True
    )

    if "题材" in duration_df.columns:

        subject_duration = duration_df.groupby(
            "题材"
        )["时长(分钟)"].mean().reset_index()

        fig_subject_duration = px.bar(
            subject_duration.sort_values(
                by="时长(分钟)",
                ascending=False
            ),
            x="题材",
            y="时长(分钟)",
            color="时长(分钟)",
            title="不同历史题材的平均视频时长"
        )

        fig_subject_duration.update_layout(
            xaxis_tickangle=-35,
            paper_bgcolor="#f6f1e7",
            plot_bgcolor="#fffdf7"
        )

        st.plotly_chart(
            fig_subject_duration,
            use_container_width=True
        )

    best_duration = duration_group.sort_values(
        by="平均播放数",
        ascending=False
    ).iloc[0]

    short_ratio = (duration_df["时长(秒)"] < 180).mean()
    long_ratio = (duration_df["时长(秒)"] >= 600).mean()

    if short_ratio > long_ratio:

        duration_conclusion = f"""
        数据显示，当前样本中短时长历史视频占比较高，3分钟以内视频占比为{short_ratio:.2%}。从传播结果看，"{best_duration['时长类型']}"视频的平均播放表现最突出。这说明平台上的历史内容并不只是依赖知识密度获得关注，更依赖节奏压缩、叙事钩子和情绪启动来完成快速传播。
        在这种时长结构下，历史叙事往往呈现出：
        - 背景交代被压缩
        - 人物与事件冲突被前置
        - 复杂因果被改写成更短的情节链条
        - 观看完成率成为影响传播的重要变量
        因此，时长并不是单纯的形式指标，而是平台历史叙事被重新组织的重要机制。
        """

    else:

        duration_conclusion = f"""
        数据显示，当前样本中长时长历史视频仍然具有较强存在感，10分钟以上视频占比为{long_ratio:.2%}。从传播结果看，"{best_duration['时长类型']}"视频的平均播放表现最突出。这说明历史内容在平台传播中仍保留一定的解释空间，观众并非只接受碎片化内容，也会为具有叙事张力和知识密度的视频投入较长观看时间。

        但长视频若要获得传播，通常仍需要在开头建立明确冲突、悬念或情绪入口，否则容易在平台推荐机制中失去竞争力。
        """

    st.markdown(
        f'<div class="conclusion-box">{duration_conclusion}</div>',
        unsafe_allow_html=True
    )

# =====================================
# 情绪分析
# =====================================

st.header("历史叙事中的情绪传播")

fig2 = px.histogram(
    df,
    x="自动情绪得分",
    nbins=30,
    title="历史视频情绪得分分布"
)

fig2.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# 自动生成结论
# =====================================

high_emotion_ratio = (
    df["自动情绪得分"] > 0.7
).mean()

low_emotion_ratio = (
    df["自动情绪得分"] < 0.3
).mean()

st.subheader("课堂思考")

if high_emotion_ratio > 0.4:

    emotion_conclusion = """
    数据显示，大量高传播历史视频呈现出明显的高情绪化特征。这意味着平台传播越来越依赖“高唤醒情绪”推动历史内容扩散。历史叙事正在从传统教材中的“结构解释”模式，逐渐转向以冲突、热血、阴谋与人物对立为核心的“情绪驱动”模式。在这一过程中：
    - 人物冲突被强化
    - 制度分析被压缩
    - 情绪刺激替代了复杂解释

    平台算法实际上正在重新塑造公众理解历史的方式。
    """

else:

    emotion_conclusion = """
    当前历史视频整体情绪化程度相对有限。

    历史传播仍然保留一定的知识解释功能。

    但从高播放视频来看，情绪化表达依然明显强于普通内容。
    """

st.markdown(
    f'<div class="conclusion-box">{emotion_conclusion}</div>',
    unsafe_allow_html=True
)

# =====================================
# 题材分析
# =====================================

if "题材" in df.columns:

    st.header("历史题材传播结构")

    subject_group = df.groupby(
        "题材"
    )["播放数"].mean().reset_index()

    fig3 = px.bar(
        subject_group,
        x="题材",
        y="播放数",
        color="播放数",
        title="不同历史题材平均播放量"
    )

    fig3.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # 自动分析题材

    top_subject = subject_group.sort_values(
        by="播放数",
        ascending=False
    ).iloc[0]["题材"]

    st.markdown(
        f'''
        <div class="conclusion-box">
        数据显示，"{top_subject}"类历史题材拥有最强传播能力。

        这说明平台更偏好：

        - 冲突性强
        - 人物对立明显
        - 情绪张力高
        - 叙事节奏快

        的历史内容。

        相比之下，制度史、经济史等需要长期解释链条支撑的内容，更难获得高传播。

        平台传播逻辑正在推动历史叙事从“结构化历史”向“戏剧化历史”转变。
        ''',
        unsafe_allow_html=True
    )

# =====================================
# UP主分析
# =====================================

st.header("历史传播中的头部控制现象")

up_stats = df.groupby("UP主").agg({

    "播放数": ["mean", "sum", "count"],
    "点赞数": "mean"

})

up_stats.columns = [

    "平均播放",
    "总播放",
    "视频数量",
    "平均点赞"

]

up_stats = up_stats.reset_index()

# TOP10占比

total_play = up_stats["总播放"].sum()

up_stats["播放占比"] = (
    up_stats["总播放"] /
    total_play
)


top10_ratio = up_stats.sort_values(
    by="总播放",
    ascending=False
).head(10)["播放占比"].sum()

st.metric(
    "TOP10 UP主播放量占比",
    f"{top10_ratio:.2%}"
)

fig4 = px.bar(
    up_stats.sort_values(
        by="总播放",
        ascending=False
    ).head(20),

    x="UP主",
    y="总播放",

    color="视频数量",

    title="TOP20历史区UP主"
)

fig4.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# 自动生成UP主结论

if top10_ratio > 0.5:

    up_conclusion = f"""
    数据显示，TOP10历史UP主占据了{top10_ratio:.2%}的总播放量。这意味着历史传播存在明显的“头部控制现象”。
    少数高流量创作者实际上正在主导公众的历史认知。
    平台历史叙事并非完全自由竞争，而是呈现出明显的注意力集中结构。
    这种结构会进一步强化：
    - 爽文化叙事
    - 情绪化表达
    - 高冲突历史内容
    因为头部UP主更倾向于生产符合平台算法偏好的内容。
    """

else:

    up_conclusion = """
    历史区整体传播结构相对分散。

    不同类型创作者仍然能够获得一定传播空间。
    """

st.markdown(
    f'<div class="conclusion-box">{up_conclusion}</div>',
    unsafe_allow_html=True
)

# =====================================
# 聚类分析
# =====================================

st.header("历史叙事聚类结构")

features = up_stats[[
    "平均播放",
    "视频数量",
    "平均点赞"
]]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

up_stats["聚类类别"] = kmeans.fit_predict(
    scaled_features
)

fig5 = px.scatter(

    up_stats,

    x="视频数量",
    y="平均播放",

    color=up_stats["聚类类别"].astype(str),

    size="总播放",

    hover_data=["UP主"],

    title="历史UP主聚类结构"
)

fig5.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# =====================================
# 词云
# =====================================

st.header("高频历史叙事词汇")

text = " ".join(
    df["标题"].astype(str)
)

words = jieba.lcut(text)

stopwords = [
    "中国",
    "历史",
    "为什么",
    "如何",
    "一个"
]

filtered_words = [
    w for w in words
    if w not in stopwords and len(w) > 1
]

final_text = " ".join(filtered_words)

wc = WordCloud(

    font_path="STSONG.TTF",

    width=1200,
    height=600,

    background_color="white"

).generate(final_text)

fig6, ax6 = plt.subplots(
    figsize=(15,8)
)

ax6.imshow(wc)

ax6.axis("off")

st.pyplot(fig6)

# =====================================
# 情绪聚类分析
# =====================================

if "情绪" in df.columns:

    st.header("不同情绪类型的传播聚类")

    emotion_cluster = df.groupby("情绪").agg({
        "播放数": "mean",
        "互动率": "mean",
        "点赞数": "mean",
        "收藏": "mean"
    }).reset_index()

    fig_emotion = px.scatter(
        emotion_cluster,
        x="互动率",
        y="播放数",
        size="点赞数",
        color="情绪",
        hover_data=["收藏"],
        title="不同情绪的传播聚类结构"
    )

    fig_emotion.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig_emotion,
        use_container_width=True
    )

    st.markdown("""
    <div class="conclusion-box">
    从情绪聚类结果来看：

    - 高情绪内容往往拥有更高互动率
    - 愤怒、震撼、热血等情绪更容易获得传播
    - 理性解释类内容虽然稳定，但爆发力较弱

    说明平台传播逻辑更偏向“情绪唤醒”，而非“理性说服”。

    在平台机制影响下，历史内容越来越接近短视频娱乐逻辑。
    """, unsafe_allow_html=True)

# =====================================
# 题材聚类分析
# =====================================

if "题材" in df.columns:

    st.header("历史题材聚类结构")

    subject_cluster = df.groupby("题材").agg({
        "播放数": "mean",
        "互动率": "mean",
        "点赞数": "mean",
        "收藏": "mean"
    }).reset_index()

    fig_subject = px.scatter(
        subject_cluster,
        x="互动率",
        y="播放数",
        size="点赞数",
        color="题材",
        hover_data=["收藏"],
        title="不同历史题材传播聚类"
    )

    fig_subject.update_layout(
        paper_bgcolor="#f6f1e7",
        plot_bgcolor="#fffdf7"
    )

    st.plotly_chart(
        fig_subject,
        use_container_width=True
    )

    st.markdown("""
    <div class="conclusion-box">
    不同历史题材之间存在明显传播差异。

    战争史、帝王史、政治斗争类内容更容易形成高传播。

    而制度史、经济史、社会史等强调长期结构分析的内容传播相对较弱。

    这意味着：

    平台算法更偏好“戏剧性历史”，而不是“结构性历史”。

    历史传播正在从传统教材中的“宏观解释”，逐渐转向短视频中的“事件冲突”。
    """, unsafe_allow_html=True)

# =====================================
# 互动率排行
# =====================================

st.header("高互动率视频")

interaction_top = df.sort_values(
    by="互动率",
    ascending=False
).head(20)

st.dataframe(
    interaction_top[[
        "标题",
        "UP主",
        "播放数",
        "点赞数",
        "互动率"
    ]]
)

fig_interaction = px.bar(
    interaction_top,
    x="标题",
    y="互动率",
    color="播放数",
    hover_data=["UP主"],
    title="高互动率视频排行"
)

fig_interaction.update_layout(
    xaxis_tickangle=-45,
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig_interaction,
    use_container_width=True
)

# =====================================
# TOP视频分析
# =====================================

st.header("高播放历史视频结构")

high_play = df.sort_values(
    by="播放数",
    ascending=False
).head(30)

fig_top = px.scatter(
    high_play,
    x="播放数",
    y="点赞数",
    size="收藏",
    color="题材" if "题材" in df.columns else None,
    hover_data=["标题", "UP主"],
    title="高播放历史视频传播结构"
)

fig_top.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig_top,
    use_container_width=True
)

st.markdown("""
<div class="conclusion-box">
高播放历史视频普遍呈现出：
- 强情绪标题
- 人物冲突叙事
- 极端化表达
- 高频悬念结构
说明短视频时代的历史传播，已经逐渐形成类似“爽文”的叙事逻辑。
历史不再只是知识内容，而正在成为一种情绪消费品。
""", unsafe_allow_html=True)

# =====================================
# 词频统计
# =====================================

st.header("高频历史词汇统计")

from collections import Counter

word_count = Counter(filtered_words)

word_df = pd.DataFrame(
    word_count.items(),
    columns=["词语", "频率"]
).sort_values(by="频率", ascending=False).head(20)

fig_word = px.bar(
    word_df,
    x="词语",
    y="频率",
    color="频率",
    title="历史高频词汇"
)

fig_word.update_layout(
    paper_bgcolor="#f6f1e7",
    plot_bgcolor="#fffdf7"
)

st.plotly_chart(
    fig_word,
    use_container_width=True
)

st.markdown("""
<div class="conclusion-box">
高频词汇中大量出现：

- 皇帝
- 战争
- 崛起
- 崩溃
- 阴谋
- 王朝

说明当前历史传播高度集中于：

“权力斗争型叙事”。

而普通社会、经济结构、民众生活等内容相对缺席。

平台传播逻辑强化了历史中的戏剧冲突部分。
""", unsafe_allow_html=True)

# =====================================
# 下载结果
# =====================================

st.header("研究数据导出")

csv = up_stats.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="下载UP主聚类结果",
    data=csv,
    file_name="历史传播聚类结果.csv",
    mime="text/csv"
)

# =====================================
# 总结
# =====================================

st.header("本课小结")

st.markdown("""
<div class="conclusion-box">
本研究发现：

1. 平台传播正在推动历史叙事向情绪化、冲突化方向演化。

2. 高传播历史内容越来越依赖人物对立与情绪刺激，而非制度解释。

3. 平台算法并未消灭历史知识，而是重新定义了“什么样的历史更容易被看见”。

4. 历史传播呈现明显头部集中现象，少数UP主对公众历史认知具有重要影响。

5. 数字平台不仅改变了传播方式，也正在重构公众理解历史的方式。
""", unsafe_allow_html=True)
