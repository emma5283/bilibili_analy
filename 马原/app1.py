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

df = df.rename(
    columns={
        "up主": "UP主",
        "up主id": "UP主id"
    }
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
        数据显示，当前样本中长时长历史视频仍然具有较强存在感，10分钟以上视频占比为{long_ratio:.2%}。从传播结果看，"{best_duration['时长类型']}"视频的平均播放表现最突出。这说明历史内容在平台传播中仍保留一定的解释空间，观众并非只接受碎片化内容，也会为具有叙事张力和知识密度的视频投入较长观看时间。但长视频若要获得传播，通常仍需要在开头建立明确冲突、悬念或情绪入口，否则容易在平台推荐机制中失去竞争力。
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
    可以发现，前5大UP几乎是断层的存在。进一步分析发现，这几大UP主都展现出了多元化的叙事风格。
    燕三嘤嘤嘤为时事分析博主，主要内容围绕当下国际形势展开，内容相对客观，语言较为活泼。
    小哈哈里Harry视频则主要聚焦于历史上的猎奇事件，并利用夸张的标题吸引关注，获得了相当高的播放量。
    小约翰可汗则主要聚焦于历史上的“硬核狠人”“神奇组织”，利用较长的视频向观众科普一些历史事件，但与小哈哈里相比，其视频时长更长，几乎全部超出30分钟，并且视频标题的猎奇性低于小哈哈里。
    古人云的形式则更加多元与不同，漫画的表现形式、人物的鬼畜表情，和题材的丰富多样，让他抓住了“短平快”的流量叙事密码，可以发现，其视频时长几乎不超过五分钟。
    而史界巡游官，可以说是一个相当奇葩的存在，其视频几乎全为ai制作，可以看出明显的穿模与前后逻辑不连贯，标题也十分简单，但视频内容十分猎奇。点开其高播放量视频，几乎满屏都是问号。
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
# 视角、情绪、映射聚类分析
# =====================================

cluster_dimensions = [
    col for col in ["视角", "情绪", "映射"]
    if col in df.columns
]

if cluster_dimensions:

    st.header("视角、情绪与映射的传播聚类")

    cluster_tables = []

    for dimension in cluster_dimensions:

        dimension_stats = df.dropna(
            subset=[dimension, "播放数", "互动率"]
        ).groupby(dimension).agg({
            "播放数": "mean",
            "互动率": "mean",
            "点赞数": "mean",
            "收藏": "mean",
            "转发": "mean",
            "标题": "count"
        }).reset_index()

        dimension_stats.columns = [
            "类型",
            "平均播放数",
            "平均互动率",
            "平均点赞数",
            "平均收藏数",
            "平均转发数",
            "视频数量"
        ]

        dimension_stats["分析维度"] = dimension

        if len(dimension_stats) >= 2:

            cluster_features = dimension_stats[[
                "平均播放数",
                "平均互动率",
                "平均点赞数",
                "平均收藏数",
                "平均转发数",
                "视频数量"
            ]].fillna(0)

            scaled_cluster_features = StandardScaler().fit_transform(
                cluster_features
            )

            cluster_count = min(3, len(dimension_stats))

            dimension_kmeans = KMeans(
                n_clusters=cluster_count,
                random_state=42,
                n_init=10
            )

            dimension_stats["聚类类别"] = dimension_kmeans.fit_predict(
                scaled_cluster_features
            )

        else:

            dimension_stats["聚类类别"] = 0

        cluster_tables.append(dimension_stats)

    narrative_cluster = pd.concat(
        cluster_tables,
        ignore_index=True
    )

    tab_view, tab_emotion, tab_mapping, tab_total = st.tabs([
        "视角",
        "情绪",
        "映射",
        "综合对比"
    ])

    dimension_tabs = {
        "视角": tab_view,
        "情绪": tab_emotion,
        "映射": tab_mapping
    }

    for dimension, tab in dimension_tabs.items():

        with tab:

            if dimension not in cluster_dimensions:

                st.info(f"当前数据中没有“{dimension}”列。")
                continue

            dimension_data = narrative_cluster[
                narrative_cluster["分析维度"] == dimension
            ].copy()

            top_play_type = dimension_data.sort_values(
                by="平均播放数",
                ascending=False
            ).iloc[0]

            top_interaction_type = dimension_data.sort_values(
                by="平均互动率",
                ascending=False
            ).iloc[0]

            col_a, col_b, col_c = st.columns(3)

            col_a.metric(
                "类型数量",
                len(dimension_data)
            )

            col_b.metric(
                "最高平均播放类型",
                top_play_type["类型"],
                f"{top_play_type['平均播放数']:,.0f}"
            )

            col_c.metric(
                "最高平均互动类型",
                top_interaction_type["类型"],
                f"{top_interaction_type['平均互动率']:.2%}"
            )

            fig_dimension_cluster = px.scatter(
                dimension_data,
                x="平均互动率",
                y="平均播放数",
                size="视频数量",
                color=dimension_data["聚类类别"].astype(str),
                hover_data=[
                    "类型",
                    "平均点赞数",
                    "平均收藏数",
                    "平均转发数",
                    "视频数量"
                ],
                text="类型",
                title=f"{dimension}类型的播放量与互动率聚类"
            )

            fig_dimension_cluster.update_traces(
                textposition="top center"
            )

            fig_dimension_cluster.update_layout(
                paper_bgcolor="#f6f1e7",
                plot_bgcolor="#fffdf7"
            )

            st.plotly_chart(
                fig_dimension_cluster,
                use_container_width=True
            )

            fig_dimension_bar = px.bar(
                dimension_data.sort_values(
                    by="平均播放数",
                    ascending=False
                ),
                x="类型",
                y="平均播放数",
                color="平均互动率",
                text="视频数量",
                title=f"不同{dimension}类型的平均播放量"
            )

            fig_dimension_bar.update_layout(
                xaxis_tickangle=-35,
                paper_bgcolor="#f6f1e7",
                plot_bgcolor="#fffdf7"
            )

            st.plotly_chart(
                fig_dimension_bar,
                use_container_width=True
            )

            st.dataframe(
                dimension_data[[
                    "类型",
                    "聚类类别",
                    "视频数量",
                    "平均播放数",
                    "平均互动率",
                    "平均点赞数",
                    "平均收藏数",
                    "平均转发数"
                ]].sort_values(
                    by="平均播放数",
                    ascending=False
                ),
                use_container_width=True
            )

    with tab_total:

        fig_total_cluster = px.scatter(
            narrative_cluster,
            x="平均互动率",
            y="平均播放数",
            size="视频数量",
            color="分析维度",
            symbol=narrative_cluster["聚类类别"].astype(str),
            hover_data=[
                "类型",
                "聚类类别",
                "平均点赞数",
                "平均收藏数",
                "平均转发数",
                "视频数量"
            ],
            title="视角、情绪与映射类型的综合传播对比"
        )

        fig_total_cluster.update_layout(
            paper_bgcolor="#f6f1e7",
            plot_bgcolor="#fffdf7"
        )

        st.plotly_chart(
            fig_total_cluster,
            use_container_width=True
        )

        st.dataframe(
            narrative_cluster.sort_values(
                by="平均播放数",
                ascending=False
            ),
            use_container_width=True
        )

    strongest_play = narrative_cluster.sort_values(
        by="平均播放数",
        ascending=False
    ).iloc[0]

    strongest_interaction = narrative_cluster.sort_values(
        by="平均互动率",
        ascending=False
    ).iloc[0]

    st.markdown(
        f"""
        <div class="conclusion-box">
        从视角、情绪与映射三类叙事标签的聚类结果看，平台传播并不是平均分配给所有表达方式，而是更集中地流向少数高识别度类型。

        平均播放量最高的是“{strongest_play['分析维度']}”维度中的“{strongest_play['类型']}”，平均播放量约为{strongest_play['平均播放数']:,.0f}；平均互动率最高的是“{strongest_interaction['分析维度']}”维度中的“{strongest_interaction['类型']}”，平均互动率约为{strongest_interaction['平均互动率']:.2%}。

        这说明历史视频的传播优势往往来自三种机制的叠加：视角上容易让观众迅速代入，情绪上能够形成明确态度，映射上能够把历史事件与现实经验连接起来。播放量更高的类型负责扩大触达，互动率更高的类型则更容易激发评论、点赞、收藏和转发。
        </div>
        """,
        unsafe_allow_html=True
    )

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
