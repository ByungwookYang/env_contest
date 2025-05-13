import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import streamlit as st
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform


import matplotlib.font_manager as fm
import os

# 폰트 경로 지정
font_path = "assets/fonts/NanumGothic-Regular.ttf"

if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rc("font", family="sans-serif")  # fallback

# 마이너스 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


def show_tab4_seasonal_keywords():
    st.header("🌸 계절별 키워드 시각화")
    df_bow = pd.read_csv("df_bow.csv")
    X_words = df_bow.T
    X_words.index = pd.Index([str(w).strip() for w in X_words.index])  # 문자열 정리
    words = np.array(X_words.index)

    # 2. 거리 계산 (Jaccard)
    dist_array = pdist(X_words, metric="jaccard")
    dist_matrix = squareform(dist_array)

    # 3. t-SNE
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=2,
        random_state=42,
        init="random",
    )
    X_tsne = tsne.fit_transform(dist_matrix)

    dfdfdf = pd.read_csv("dfdfdf.csv")

    # 1. 계절별 단어 리스트 저장
    season_word_dict = {}
    for season in dfdfdf["season"].unique():
        season_df = dfdfdf[dfdfdf["season"] == season]
        word_df = season_df.drop(columns=["season"])
        appeared_words = word_df.columns[(word_df != 0).any()].tolist()
        season_word_dict[season] = set(appeared_words)

    # 2. 모든 단어의 등장 계절 수 계산
    word_counter = Counter()
    for words in season_word_dict.values():
        word_counter.update(words)

    # 3. 두 계절 이상에 등장한 단어는 제거
    common_words = set([word for word, count in word_counter.items() if count >= 2])
    for season in season_word_dict:
        season_word_dict[season] = sorted(season_word_dict[season] - common_words)

    # 단어 인덱스 정제
    X_words.index = pd.Index([str(w).strip() for w in X_words.index])
    words = np.array(X_words.index)
    X_tsne = X_tsne[: len(words)]

    # season_word_dict 정제
    season_word_dict = {
        season: {str(w).strip() for w in words_set}
        for season, words_set in season_word_dict.items()
    }

    season_colors = {"봄": "red", "여름": "green", "가을": "orange", "겨울": "blue"}

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, season in enumerate(["봄", "여름", "가을", "겨울"]):
        ax = axes[i]
        highlight_words = season_word_dict.get(season, set())

        mask = np.array([w in highlight_words for w in words])
        X_highlight = X_tsne[mask]
        X_others = X_tsne[~mask]
        words_highlight = words[mask]
        words_others = words[~mask]

        ax.scatter(X_others[:, 0], X_others[:, 1], color="lightgray", alpha=0.6)
        for j, w in enumerate(words_others):
            ax.text(
                X_others[j, 0],
                X_others[j, 1],
                w,
                fontsize=8,
                color="gray",
                alpha=0.4,
                fontproperties=font_prop,  # ✅ 폰트 적용
            )

        ax.scatter(
            X_highlight[:, 0], X_highlight[:, 1], color=season_colors[season], alpha=0.9
        )
        for j, w in enumerate(words_highlight):
            ax.text(
                X_highlight[j, 0],
                X_highlight[j, 1],
                w,
                fontsize=10,
                color=season_colors[season],
                weight="bold",
                fontproperties=font_prop,  # ✅ 폰트 적용
            )

        ax.set_title(
            f"{season} 키워드 시각화",
            fontsize=14,
            fontproperties=font_prop,  # ✅ 제목에도 폰트 적용
        )

    plt.tight_layout()
    st.pyplot(fig)

    # 🔽 계절별 키워드 출력
    st.subheader("📝 계절별 고유 키워드 목록")
    for season in ["봄", "여름", "가을", "겨울"]:
        keywords = sorted(season_word_dict.get(season, []))
        st.markdown(f"**{season}** ({len(keywords)}개 단어):")
        st.markdown(", ".join(keywords) if keywords else "_(고유 단어 없음)_")
        st.markdown("---")
