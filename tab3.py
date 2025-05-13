import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def show_tab3_injury_keywords():
    st.header("📍 중요 키워드 강조 시각화")
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
    X_words.index = pd.Index([str(w).strip() for w in X_words.index])
    words = np.array(X_words.index)
    X_tsne = X_tsne[: len(words)]

    # 🔴 강조할 주요 키워드 리스트
    highlight_words = [
        "작업",
        "보관",
        "사고",
        "배관",
        "차량",
        "압력",
        "운반",
        "폭발",
        "이송",
        "암모니아",
        "유출",
        "공급",
        "공정",
        "밸브",
        "탱크",
        "잔류",
        "수산화나트륨",
        "저장",
        "가동",
        "냉동",
        "주입",
        "교체",
        "노후",
        "호스",
        "제품",
        "드럼",
        "투입",
        "개방",
        "원료",
        "현장",
    ]

    # ✅ 마스크 생성
    mask = np.array([w in highlight_words for w in words])

    X_highlight = X_tsne[mask]
    X_others = X_tsne[~mask]
    words_highlight = np.array(words)[mask]
    words_others = np.array(words)[~mask]

    # 📊 시각화
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        X_others[:, 0], X_others[:, 1], color="black", alpha=0.6, label="기타 단어"
    )
    for i, w in enumerate(words_others):
        ax.text(X_others[i, 0], X_others[i, 1], w, fontsize=8, color="black")

    ax.scatter(
        X_highlight[:, 0], X_highlight[:, 1], color="red", alpha=0.9, label="중요 단어"
    )
    for i, w in enumerate(words_highlight):
        ax.text(X_highlight[i, 0], X_highlight[i, 1], w, fontsize=8, color="red")

    ax.set_title("중요 키워드 강조 시각화 (t-SNE 기반)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)

    # 📌 강조된 중요 키워드 목록도 함께 표시
    st.markdown("### 🧩 강조된 중요 키워드")
    st.markdown(", ".join(highlight_words))
