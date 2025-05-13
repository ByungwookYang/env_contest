import streamlit as st
from modules.handler import stream_handler
from bk_st_function import print_messages, add_message


def show_tab1_chatbot():
    st.subheader("💡 예시 질문")
    st.markdown(
        """
        - **우리 사업장에서 발생할 수 있는 사고나 위험은 뭐가 있을까요?**
        - **계절에 따라 주의해야 할 사고 유형이 있을까요?**
        - **작업자나 장비 측면에서 어떤 점이 특히 위험할까요?**
        - **시설이 노후되었는데, 그럴 때 어떤 사고가 자주 발생하나요?**
        - **운반이나 보관 중에 주의할 점은 무엇인가요?**
        """
    )
    # ✅ tab1일 때만 입력창 노출
    user_input = st.chat_input("궁금한 내용을 물어보세요!")

    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    # 이전 대화 기록 출력
    print_messages()

    # 만약에 사용자 입력이 들어오면...
    if user_input:
        agent = st.session_state["react_agent"]
        # Config 설정

        if agent is not None:
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            # 사용자의 입력
            st.chat_message("user").write(user_input)

            with st.chat_message("assistant"):
                # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
                container = st.empty()

                ai_answer = ""
                container_messages, tool_args, agent_answer = stream_handler(
                    container,
                    agent,
                    {
                        "messages": [
                            ("human", user_input),
                        ]
                    },
                    config,
                )

                # 대화기록을 저장한다.
                add_message("user", user_input)
                for tool_arg in tool_args:
                    add_message(
                        "assistant",
                        tool_arg["tool_result"],
                        "tool_result",
                        tool_arg["tool_name"],
                    )
                add_message("assistant", agent_answer)
        else:
            warning_msg.warning("사이드바에서 사업장의 특징을 입력해주세요.")
