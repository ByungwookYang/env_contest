import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import streamlit as st
from matplotlib import cm


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


def show_tab2_visualization():
    st.header("📊 키워드 시각화")

    # 1. 데이터 로딩 및 전처리
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

    # 4. 클러스터링 (KMeans)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_tsne)

    clustered_words = pd.DataFrame({"word": X_words.index, "cluster": labels})

    # 5. 클러스터 주제명 사전
    cluster_titles = {
        0: "반응성 물질 및 물리적 조건에 의한 사고",
        1: "폐기물 처리 및 유해 노출 사고",
        2: "폭발성 혼합물 / 공정 위험",
        3: "부식성 물질 및 설비 노후로 인한 누출 사고",
        4: "휘발성 유기화합물 증기 노출 및 정전기 폭발",
        5: "시설 누출, 미세 방출, 환경 확산형 사고",
        6: "장치 결함 및 감지 실패",
        7: "폭발·질식 유발물과 작업장 위험",
        8: "운반 중 기계적 충격 및 누출",
        9: "산화성/살균성 무기물질의 확산 위험",
    }

    # 6. 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        idx = labels == cluster_id
        ax.scatter(
            X_tsne[idx, 0],
            X_tsne[idx, 1],
            color=colors[cluster_id],
            alpha=0.7,
            label=cluster_titles[cluster_id],
        )
        for i in np.where(idx)[0]:
            ax.text(X_tsne[i, 0], X_tsne[i, 1], words[i], fontsize=9)

    legend_patches = [
        Patch(color=colors[i], label=cluster_titles[i]) for i in range(n_clusters)
    ]
    ax.legend(
        handles=legend_patches,
        title="클러스터 주제",
        bbox_to_anchor=(0.9, 1.05),
        loc="upper center",
    )

    ax.set_title("Jaccard 기반 t-SNE + KMeans 클러스터링")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    # 8. 클러스터별 키워드 출력
    st.subheader("🔍 클러스터별 키워드 목록")
    for cluster_id in range(n_clusters):
        words_in_cluster = clustered_words[clustered_words["cluster"] == cluster_id][
            "word"
        ].tolist()
        title = cluster_titles.get(cluster_id, f"Cluster {cluster_id}")
        st.markdown(f"##### - {title} ({len(words_in_cluster)} words)")
        st.markdown(", ".join(words_in_cluster))
