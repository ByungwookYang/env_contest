import streamlit as st
from bk_messages import random_uuid
from modules.agent import create_agent_executor
from dotenv import load_dotenv
from modules.tools import WebSearchTool
from tab1 import show_tab1_chatbot
from tab2 import show_tab2_visualization
from tab3 import show_tab3_injury_keywords
from tab4 import show_tab4_seasonal_keywords
from modules.handler import stream_handler
from bk_st_function import print_messages, add_message

# API KEY 정보로드
load_dotenv()

st.title("🚨사고 발생 방지 도우미🚨")

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# include_domains 초기화
if "include_domains" not in st.session_state:
    st.session_state["include_domains"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt = "화학약품을 다루는 중소형 제조업체이며, 부식성 액체와 가연성 물질을 다량 보관하고 있습니다. 탱크로리 입출고가 잦고, 현장에는 야간 교대 근무가 있으며 노후된 배관이 일부 존재합니다."
    user_text_prompt = st.text_area(
        "사업장 특징 입력", value=prompt, key="user_text_prompt"
    )

    # 설정 버튼
    apply_btn = st.button("설정 완료", type="primary")

    # 설정 버튼이 눌리면...
    if apply_btn:
        feature_message = st.empty()
        feature_message.success("사업장 특징이 입력되었습니다.")
        tool = WebSearchTool().create()
        tool.max_results = 3
        tool.include_domains = st.session_state["include_domains"]
        tool.topic = "general"
        st.session_state["react_agent"] = create_agent_executor(
            prompt,
            model_name="gpt-4o-mini",
            tools=[tool],
        )
        st.session_state["thread_id"] = random_uuid()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 사고 예방 챗봇",
        "📘 키워드 분석",
        "🚨 인명피해 키워드",
        "🌸 계절별 키워드 시각화",
    ]
)

with tab1:
    show_tab1_chatbot()

with tab2:
    show_tab2_visualization()

with tab3:
    show_tab3_injury_keywords()

with tab4:
    show_tab4_seasonal_keywords()


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = random_uuid()
