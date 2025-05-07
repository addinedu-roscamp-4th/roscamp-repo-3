import re
import streamlit as st
from typing import Dict, List, Optional, Any

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, tool

# 음성 관련 임포트
import whisper
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import io

# DB 연결을 위한 임포트 (실제 구현 시 필요)
# from sqlalchemy import create_engine, Column, Integer, String, Float
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# 메뉴 및 옵션 데이터 (실제로는 DB에서 가져올 수 있음)
MENU_DATA = {
    "불고기 샌드위치": {
        "price": 6500,
        "description": "부드러운 불고기가 들어간 샌드위치",
    },
    "새우 샌드위치": {"price": 6200, "description": "신선한 새우가 들어간 샌드위치"},
    "베이컨 샌드위치": {
        "price": 6000,
        "description": "바삭한 베이컨이 들어간 샌드위치",
    },
}

SAUCE_DATA = {
    "이탈리안": {"description": "이탈리안 스타일의 소스"},
    "칠리": {"description": "매콤한 칠리 소스"},
}

VEGETABLE_DATA = {
    "양상추": {"price": 0, "description": "기본 제공되는 양상추"},
    "로메인": {"price": 700, "description": "신선한 로메인 (+700원)"},
    "바질": {"price": 800, "description": "향긋한 바질 (+800원)"},
}

CHEESE_DATA = {
    "슬라이스 치즈": {"price": 0, "description": "기본 제공되는 슬라이스 치즈"},
    "슈레드 치즈": {"price": 1000, "description": "풍부한 슈레드 치즈 (+1000원)"},
    "모짜렐라 치즈": {"price": 1300, "description": "쫄깃한 모짜렐라 치즈 (+1300원)"},
}


# 주문 상태를 저장할 클래스
class OrderState:
    def __init__(self):
        self.menu = None
        self.sauce = None
        self.vegetable = "양상추"
        self.cheese = "슬라이스 치즈"
        self.step = "menu"
        self.confirmed = False

    def get_dict(self):
        return {
            "menu": self.menu,
            "sauce": self.sauce,
            "vegetable": self.vegetable,
            "cheese": self.cheese,
            "step": self.step,
            "confirmed": self.confirmed,
        }

    def reset(self):
        self.__init__()


# ====== Tool 정의 ======


@tool
def get_menu_list(query: str = "") -> str:
    """
    메뉴 목록과 가격을 조회합니다.

    Args:
        query: 검색어 (선택적)

    Returns:
        메뉴 목록 및 가격 정보
    """
    result = "메뉴 목록:\n"
    for name, info in MENU_DATA.items():
        if query and query not in name:
            continue
        result += f"- {name}: {info['price']}원 - {info['description']}\n"
    return result


@tool
def get_sauce_list(query: str = "") -> str:
    """
    소스 목록을 조회합니다.

    Args:
        query: 검색어 (선택적)

    Returns:
        소스 목록 정보
    """
    result = "소스 목록:\n"
    for name, info in SAUCE_DATA.items():
        if query and query not in name:
            continue
        result += f"- {name}: {info['description']}\n"
    return result


@tool
def get_vegetable_list(query: str = "") -> str:
    """
    야채 목록과 추가 가격을 조회합니다.

    Args:
        query: 검색어 (선택적)

    Returns:
        야채 목록 및 가격 정보
    """
    result = "야채 목록:\n"
    for name, info in VEGETABLE_DATA.items():
        if query and query not in name:
            continue
        result += f"- {name}: {info['description']}\n"
    return result


@tool
def get_cheese_list(query: str = "") -> str:
    """
    치즈 목록과 추가 가격을 조회합니다.

    Args:
        query: 검색어 (선택적)

    Returns:
        치즈 목록 및 가격 정보
    """
    result = "치즈 목록:\n"
    for name, info in CHEESE_DATA.items():
        if query and query not in name:
            continue
        result += f"- {name}: {info['description']}\n"
    return result


@tool
def update_order(
    menu: Optional[str] = None,
    sauce: Optional[str] = None,
    vegetable: Optional[str] = None,
    cheese: Optional[str] = None,
) -> str:
    """
    주문 정보를 업데이트합니다.

    Args:
        menu: 선택한 메뉴 이름
        sauce: 선택한 소스 이름
        vegetable: 선택한 야채 이름
        cheese: 선택한 치즈 이름

    Returns:
        업데이트된 주문 정보 요약
    """
    order_state = st.session_state.order_state

    if menu and menu in MENU_DATA:
        order_state.menu = menu
        order_state.step = "sauce" if order_state.step == "menu" else order_state.step

    if sauce and sauce in SAUCE_DATA:
        order_state.sauce = sauce
        order_state.step = (
            "vegetable" if order_state.step == "sauce" else order_state.step
        )

    if vegetable and vegetable in VEGETABLE_DATA:
        order_state.vegetable = vegetable
        order_state.step = (
            "cheese" if order_state.step == "vegetable" else order_state.step
        )

    if cheese and cheese in CHEESE_DATA:
        order_state.cheese = cheese
        order_state.step = (
            "confirm" if order_state.step == "cheese" else order_state.step
        )

    return get_order_summary()


@tool
def get_order_summary() -> str:
    """
    현재 주문 요약 정보를 반환합니다.

    Returns:
        주문 요약 및 가격 정보
    """
    order_state = st.session_state.order_state

    if not order_state.menu:
        return "아직 메뉴를 선택하지 않았습니다."

    base_price = MENU_DATA[order_state.menu]["price"] if order_state.menu else 0
    veg_price = (
        VEGETABLE_DATA[order_state.vegetable]["price"] if order_state.vegetable else 0
    )
    cheese_price = CHEESE_DATA[order_state.cheese]["price"] if order_state.cheese else 0
    total = base_price + veg_price + cheese_price

    summary = (
        f"메뉴: {order_state.menu} ({base_price}원)\n"
        f"소스: {order_state.sauce}\n"
        f"야채: {order_state.vegetable} (+{veg_price}원)\n"
        f"치즈: {order_state.cheese} (+{cheese_price}원)\n"
        f"총 결제 금액: {total}원"
    )

    return summary


@tool
def confirm_order(confirm: bool) -> str:
    """
    주문을 확정하거나 취소합니다.

    Args:
        confirm: True면 주문 확정, False면 주문 취소

    Returns:
        주문 처리 결과 메시지
    """
    order_state = st.session_state.order_state

    if confirm:
        order_state.confirmed = True
        # 여기에 DB 저장 로직 추가 가능
        return f"✅ 주문이 완료되었습니다!\n{get_order_summary()}"
    else:
        order_state.reset()
        return "🔄 주문을 처음부터 다시 시작합니다."


@tool
def speech_to_text() -> str:
    """
    음성을 텍스트로 변환합니다.

    Returns:
        변환된 텍스트
    """
    st.info("말씀해주세요...", icon="🎤")
    sd.default.samplerate = 16000
    sd.default.channels = 1
    recording = sd.rec(int(5 * 16000))
    sd.wait()

    wav_path = "temp_whisper.wav"
    sf.write(wav_path, recording, 16000)

    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language="ko")

    return result["text"]


# ====== Agent 구성 ======


def initialize_agent():
    # 도구 목록 정의
    tools = [
        get_menu_list,
        get_sauce_list,
        get_vegetable_list,
        get_cheese_list,
        update_order,
        get_order_summary,
        confirm_order,
        speech_to_text,
    ]

    # 시스템 프롬프트 정의
    system_prompt = """
    당신은 서보웨이 무인 샌드위치 주문 시스템의 AI 도우미입니다.
    고객의 주문을 친절하게 도와주세요. 주문은 다음 단계로 진행됩니다:
    
    1. 메뉴 선택 (불고기/새우/베이컨 샌드위치)
    2. 소스 선택 (이탈리안/칠리)
    3. 야채 선택 (양상추/로메인/바질) - 기본은 양상추
    4. 치즈 선택 (슬라이스 치즈/슈레드 치즈/모짜렐라 치즈) - 기본은 슬라이스 치즈
    5. 주문 확인
    
    각 단계에서 적절한 도구를 사용하여 정보를 조회하고 주문을 업데이트하세요.
    고객이 주문을 완료하면 confirm_order 도구를 사용하여 주문을 확정하세요.
    
    항상 간결하고 명확하게 대화하며, 고객이 현재 어떤 단계에 있는지 알려주세요.
    """

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 언어 모델 초기화
    llm = ChatOpenAI(temperature=0)

    # Agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Agent 실행기 생성
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


# ====== Streamlit UI ======


def main():
    st.set_page_config(page_title="서보웨이 AI 주문", page_icon="🥪")
    st.title("🥪 서보웨이 AI 주문 시스템 (Agent 기반)")
    st.image("Menu.png")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(
                content="어서오세요! 서보웨이에 오신 것을 환영합니다. 메뉴를 주문해주세요"
            )
        ]

    if "order_state" not in st.session_state:
        st.session_state.order_state = OrderState()

    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()

    # 채팅 메시지 출력
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # 입력창 및 음성 버튼
    col1, col2 = st.columns([8, 1])

    with col1:
        user_input = st.chat_input("주문을 입력하세요...")

    with col2:
        if st.button("🎤", use_container_width=True):
            with st.spinner("음성 인식 중..."):
                user_input = speech_to_text()

    # 입력이 있으면 처리
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append(HumanMessage(content=user_input))

        # UI 업데이트
        with st.chat_message("user"):
            st.markdown(user_input)

        # Agent로 처리
        with st.spinner("처리 중..."):
            # Agent에 채팅 기록 전달
            result = st.session_state.agent.invoke(
                {
                    "input": user_input,
                    "chat_history": st.session_state.messages[
                        :-1
                    ],  # 방금 추가한 메시지 제외
                }
            )

            response = result["output"]

            # AI 응답 추가
            st.session_state.messages.append(AIMessage(content=response))

            # UI 업데이트
            with st.chat_message("assistant"):
                st.markdown(response)

            # TTS로 응답 읽기
            tts = gTTS(response, lang="ko")
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")


if __name__ == "__main__":
    main()
