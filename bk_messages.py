from langchain_core.messages import AIMessageChunk
from typing import Any, Dict, List, Callable
from dataclasses import dataclass
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid


def random_uuid():
    return str(uuid.uuid4())


# 도구 호출 시 실행되는 콜백 함수
def tool_callback(tool) -> None:
    print("[도구 호출]")
    print(f"Tool: {tool.get('tool')}")  # 사용된 도구의 이름을 출력합니다.
    if tool_input := tool.get("tool_input"):  # 도구에 입력된 값이 있다면
        for k, v in tool_input.items():
            print(f"{k}: {v}")  # 입력값의 키와 값을 출력합니다.
    print(f"Log: {tool.get('log')}")  # 도구 실행 로그를 출력합니다.


# 관찰 결과를 출력하는 콜백 함수
def observation_callback(observation) -> None:
    print("[관찰 내용]")
    print(f"Observation: {observation.get('observation')}")  # 관찰 내용을 출력합니다.


# 최종 결과를 출력하는 콜백 함수
def result_callback(result: str) -> None:
    print("[최종 답변]")
    print(result)  # 최종 답변을 출력합니다.


# 에이전트 콜백 함수들을 포함하는 데이터 클래스
@dataclass
class AgentCallbacks:
    tool_callback: Callable[[Dict[str, Any]], None] = tool_callback
    observation_callback: Callable[[Dict[str, Any]], None] = observation_callback
    result_callback: Callable[[str], None] = result_callback


class AgentStreamParser:

    def __init__(self, callbacks: AgentCallbacks = AgentCallbacks()):
        self.callbacks = callbacks
        self.output = None

    # 에이전트의 단계를 처리
    def process_agent_steps(self, step: Dict[str, Any]) -> None:
        """
        Args:
            step (Dict[str, Any]): 처리할 에이전트 단계 정보
        """
        if "actions" in step:
            self._process_actions(step["actions"])
        elif "steps" in step:
            self._process_observations(step["steps"])
        elif "output" in step:
            self._process_result(step["output"])

    # 에이전트의 액션들을 처리
    def _process_actions(self, actions: List[Any]) -> None:
        """
        Args:
            actions (List[Any]): 처리할 액션 리스트
        """
        for action in actions:
            if isinstance(action, (AgentAction, ToolAgentAction)) and hasattr(
                action, "tool"
            ):
                self._process_tool_call(action)

    # 도구 호출을 처리
    def _process_tool_call(self, action: Any) -> None:
        """
        Args:
            action (Any): 처리할 도구 호출 액션
        """
        tool_action = {
            "tool": getattr(action, "tool", None),
            "tool_input": getattr(action, "tool_input", None),
            "log": getattr(action, "log", None),
        }
        self.callbacks.tool_callback(tool_action)

    # 관찰 결과들을 처리
    def _process_observations(self, observations: List[Any]) -> None:
        """
        Args:
            observations (List[Any]): 처리할 관찰 결과 리스트
        """
        for observation in observations:
            observation_dict = {}
            if isinstance(observation, AgentStep):
                observation_dict["observation"] = getattr(
                    observation, "observation", None
                )
            self.callbacks.observation_callback(observation_dict)

    # 최종 결과를 처리
    def _process_result(self, result: str) -> None:
        """
        Args:
            result (str): 처리할 최종 결과
        """
        self.callbacks.result_callback(result)
        self.output = result


# Tool Message 청크를 처리하고 관리하는 클래스
class ToolChunkHandler:
    def __init__(self):
        self._reset_state()

    def _reset_state(self) -> None:
        """상태 초기화"""
        self.gathered = None
        self.first = True
        self.current_node = None
        self.current_namespace = None

    def _should_reset(self, node: str | None, namespace: str | None) -> bool:
        """상태 리셋 여부 확인"""
        # 파라미터가 모두 None인 경우 초기화하지 않음
        if node is None and namespace is None:
            return False

        # node만 설정된 경우
        if node is not None and namespace is None:
            return self.current_node != node

        # namespace만 설정된 경우
        if namespace is not None and node is None:
            return self.current_namespace != namespace

        # 둘 다 설정된 경우
        return self.current_node != node or self.current_namespace != namespace

    # 메시지 청크 처리
    def process_message(
        self,
        chunk: AIMessageChunk,
        node: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """
        Args:
            chunk: 처리할 AI 메시지 청크
            node: 현재 노드명 (선택사항)
            namespace: 현재 네임스페이스 (선택사항)
        """
        if self._should_reset(node, namespace):
            self._reset_state()

        self.current_node = node if node is not None else self.current_node
        self.current_namespace = (
            namespace if namespace is not None else self.current_namespace
        )

        self._accumulate_chunk(chunk)
        return self._display_tool_calls()

    # 청크 누적
    def _accumulate_chunk(self, chunk: AIMessageChunk) -> None:
        self.gathered = chunk if self.first else self.gathered + chunk
        self.first = False

    # 도구 호출 정보 출력
    def _display_tool_calls(self) -> None:
        if (
            self.gathered
            and not self.gathered.content
            and self.gathered.tool_call_chunks
            and self.gathered.tool_calls
        ):
            return self.gathered.tool_calls[0]["args"]
