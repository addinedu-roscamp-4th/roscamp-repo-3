# ===================== 필수 패치 =====================
import torch

torch.classes.__path__ = []

# ===================== 표준 라이브러리 =====================
import os
import json
import io
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import socket
import difflib

# ===================== 서드파티 라이브러리 =====================
# UI
import streamlit as st

# Whisper 모듈
import sounddevice as sd
import soundfile as sf
import whisper

# TTS 모델
from gtts import gTTS

# ===================== LangChain 관련 =====================
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel
from typing import List
from models import MenuItemSchema, OrderDataSchema

# ================== API Load =====================
from dotenv import load_dotenv

load_dotenv()

# ============== 프로젝트 설정 ==============
from config import KIOSK_HOST, KIOSK_PORT, WHISPER_MODEL

RECORD_SECONDS = 5


# Langsmith  연결
from langchain_teddynote import logging

logging.langsmith("kiosk_voice")


# =============== memory 설정 =============
from langgraph.checkpoint.memory import MemorySaver

# 메모리 설정
memory = MemorySaver()


# ===================== 메뉴 데이터 로드 =====================
@st.cache_data(ttl=300)
def load_menu_data(json_path: str = "menu.json") -> Dict[str, Any]:
    """JSON 파일에서 메뉴, 소스, 야채, 치즈 데이터를 읽어와 반환합니다. 5분간 캐시되어 반복 호출시 성능을 높입니다."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"메뉴 로드 실패: {str(e)}")
        return {"menu": {}, "sauce": {}, "vegetable": {}, "cheese": {}}


def initialize_session():
    """Streamlit 세션 상태에 메뉴, 소스, 야채, 치즈 데이터를 저장, 앱 시작시 반드시 호출 되어야 한다."""
    data = load_menu_data()
    st.session_state.update(
        {
            "menu_data": data.get("menu", {}),
            "sauce_data": data.get("sauce", {}),
            "vegetable_data": data.get("vegetable", {}),
            "cheese_data": data.get("cheese", {}),
        }
    )


# ================== 부분 일치 매칭 함수 ==================
def find_best_match(
    user_input: str, candidates: dict, threshold: float = 0.5
) -> Optional[str]:
    """
    사용자의 입력(user_input)과 후보 목록(candidates)에서 부분 일치 또는 유사도가 가장 높은 항목을 반환.
    threshold는 0~1 사이(0.5 이상 권장).
    """
    user_input_norm = (
        user_input.strip().lower()
    )  # 앞뒤 공백을 제거하고 소문자로 변환하여 비교가 쉽도록 한다.
    candidates_norm = {
        k.lower(): k for k in candidates.keys()
    }  # 후보군의 키도 모두 소문자로 변환해서 사전을 만듭니다.
    # 완전 일치
    if user_input_norm in candidates_norm:
        return candidates_norm[user_input_norm]
    # 부분 포함
    for norm, orig in candidates_norm.items():
        if user_input_norm in norm:
            return orig
    # 유사도 기반
    best = difflib.get_close_matches(
        user_input_norm, candidates_norm.keys(), n=1, cutoff=threshold
    )
    if best:
        return candidates_norm[best[0]]
    return None


# ================== 주문 상태 관리 클래스 ==================
class OrderState:
    def __init__(self, table_number=1):
        self.items = []  # 완료된 주문 아이템 목록
        self.menu = None
        self.sauce = None
        self.vegetable = None
        self.cheese = "슬라이스 치즈"
        self.step = "menu"
        self.confirmed = False
        self.table_number = table_number

    def _calc_item_price(self, menu_name, vegetable, cheese):
        base = st.session_state.menu_data.get(menu_name, {}).get("price", 0)
        veg = st.session_state.vegetable_data.get(vegetable, {}).get("price", 0) if vegetable else 0
        ch = st.session_state.cheese_data.get(cheese, {}).get("price", 0) if cheese else 0
        return base + veg + ch

    def save_current_item(self):
        """현재 선택을 아이템 목록에 저장하고 현재 선택을 초기화"""
        if self.menu:
            self.items.append({
                "name": self.menu,
                "price": self._calc_item_price(self.menu, self.vegetable, self.cheese),
                "qty": 1,
                "sauce": self.sauce or "",
                "vegetable": self.vegetable or "",
                "cheese": self.cheese or "슬라이스 치즈",
            })
        self.menu = None
        self.sauce = None
        self.vegetable = None
        self.cheese = "슬라이스 치즈"
        self.step = "menu"

    def get_all_items(self) -> list:
        """완료된 아이템 + 현재 진행 중인 아이템을 합쳐서 반환"""
        all_items = list(self.items)
        if self.menu:
            all_items.append({
                "name": self.menu,
                "price": self._calc_item_price(self.menu, self.vegetable, self.cheese),
                "qty": 1,
                "sauce": self.sauce or "",
                "vegetable": self.vegetable or "",
                "cheese": self.cheese or "슬라이스 치즈",
            })
        return all_items

    def get_dict(self) -> Dict[str, Any]:
        """키오스크와 호환되는 주문 데이터 포맷으로 변환"""
        return {
            "menu": self.get_all_items(),
            "table_number": self.table_number,
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        }

    def reset(self):
        self.__init__(self.table_number)


# ===================== 도구 함수들 =====================
@tool
def get_menu_list(tool_input: str = "") -> str:
    """메뉴 목록과 가격을 조회합니다."""
    result = "메뉴 목록:\n"
    for name, info in st.session_state.menu_data.items():
        if tool_input and tool_input not in name:
            continue
        result += f"- {name}: {info['price']}원 - {info.get('description', '')}\n"
    return result


@tool
def get_sauce_list(tool_input: str = "") -> str:
    """소스 목록을 조회합니다."""
    result = "소스 목록:\n"
    for name, info in st.session_state.sauce_data.items():
        if tool_input and tool_input not in name:
            continue
        result += f"- {name}: {info.get('description', '')}\n"
    return result


@tool
def get_vegetable_list(tool_input: str = "") -> str:
    """야채 목록과 추가 가격을 조회합니다."""
    result = "야채 목록:\n"
    for name, info in st.session_state.vegetable_data.items():
        if tool_input and tool_input not in name:
            continue
        result += f"- {name}: {info.get('description', '')}\n"
    return result


@tool
def get_cheese_list(tool_input: str = "") -> str:
    """치즈 목록과 추가 가격을 조회합니다."""
    result = "치즈 목록:\n"
    for name, info in st.session_state.cheese_data.items():
        if tool_input and tool_input not in name:
            continue
        result += f"- {name}: {info.get('description', '')}\n"
    return result


@tool
def update_order(
    menu: Optional[str] = None,
    sauce: Optional[str] = None,
    vegetable: Optional[str] = None,
    cheese: Optional[str] = None,
) -> str:
    """주문 정보를 업데이트하고 단계별로 진행합니다. (부분 일치 및 오류 안내 포함)"""
    order_state = st.session_state.order_state

    # 메뉴 단계
    if order_state.step == "menu" and menu:
        best_menu = find_best_match(menu, st.session_state.menu_data)
        if best_menu:
            order_state.menu = best_menu
            order_state.step = "sauce"
            return f"'{menu}'(으)로 인식된 메뉴: '{best_menu}' 선택됨.\n메뉴가 선택되었습니다. 소스를 골라주세요."
        else:
            return f"'{menu}' 메뉴는 없습니다. 다시 말씀해 주세요.\n{get_menu_list('')}"
    # 소스 단계
    if order_state.step == "sauce" and sauce:
        best_sauce = find_best_match(sauce, st.session_state.sauce_data)
        if best_sauce:
            order_state.sauce = best_sauce
            order_state.step = "vegetable"
            return f"'{sauce}'(으)로 인식된 소스: '{best_sauce}' 선택됨.\n소스가 선택되었습니다. 야채를 골라주세요."
        else:
            return (
                f"'{sauce}' 소스는 없습니다. 다시 말씀해 주세요.\n{get_sauce_list('')}"
            )
    # 야채 단계
    if order_state.step == "vegetable" and vegetable:
        best_veg = find_best_match(vegetable, st.session_state.vegetable_data)
        if best_veg:
            order_state.vegetable = best_veg
            order_state.step = "cheese"
            return f"'{vegetable}'(으)로 인식된 야채: '{best_veg}' 선택됨.\n야채가 선택되었습니다. 치즈를 골라주세요."
        else:
            return f"'{vegetable}' 야채는 없습니다. 다시 말씀해 주세요.\n{get_vegetable_list('')}"
    # 치즈 단계
    if order_state.step == "cheese" and cheese:
        best_cheese = find_best_match(cheese, st.session_state.cheese_data)
        if best_cheese:
            # 기존: 슬라이스 치즈는 안내만 하고 단계 미진행
            # 개선: 슬라이스 치즈도 선택 가능하도록
            order_state.cheese = best_cheese
            order_state.step = "confirm"
            return f"'{cheese}'(으)로 인식된 치즈: '{best_cheese}' 선택됨.\n치즈가 선택되었습니다. 주문을 확인해주세요."
        else:
            return f"'{cheese}' 치즈는 없습니다. 다시 말씀해 주세요.\n{get_cheese_list('')}"


@tool
def add_more_items(tool_input: str = "") -> str:
    """현재 선택한 메뉴를 저장하고 추가 주문을 받습니다."""
    order_state = st.session_state.order_state
    if not order_state.menu:
        return "아직 메뉴를 선택하지 않았습니다. 먼저 메뉴를 선택해주세요."
    order_state.save_current_item()
    return f"메뉴가 추가되었습니다. (현재 {len(order_state.items)}개)\n추가로 주문하실 메뉴를 말씀해주세요."


@tool
def get_order_summary(tool_input: str = "") -> str:
    """현재 주문 요약 정보를 반환합니다."""
    order_state = st.session_state.order_state
    all_items = order_state.get_all_items()
    if not all_items:
        return "아직 메뉴를 선택하지 않았습니다."
    total = 0
    lines = [f"테이블 {order_state.table_number}번 주문 내역:\n"]
    for idx, item in enumerate(all_items, 1):
        price = item["price"]
        total += price * item.get("qty", 1)
        lines.append(
            f"[{idx}] {item['name']} - 소스: {item['sauce']}, "
            f"야채: {item['vegetable']}, 치즈: {item['cheese']} "
            f"({price}원 x{item.get('qty', 1)})"
        )
    lines.append(f"\n총 결제 금액: {total}원")
    return "\n".join(lines)


@tool
def confirm_order(confirm: bool) -> str:
    """주문을 확정합니다."""
    order_state = st.session_state.order_state
    if confirm:
        # 진행 중인 현재 아이템이 있으면 항목에 저장
        if order_state.menu:
            order_state.save_current_item()
        if not order_state.items:
            return "주문할 메뉴가 없습니다. 먼저 메뉴를 선택해주세요."
        order_state.confirmed = True
        send_result = send_order_to_kiosk.invoke("")
        return f"주문이 완료되었습니다.\n{get_order_summary('')}\n(키오스크 응답: {send_result})"
    else:
        order_state.reset()
        return "🔄 주문을 처음부터 다시 시작합니다."


@tool
def send_order_to_kiosk() -> str:
    """주문 정보를 키오스크로 전송합니다."""
    order_state = st.session_state.order_state
    order_data = order_state.get_dict()
    if not order_data["menu"]:
        return "아직 메뉴를 선택하지 않았습니다."

    try:
        with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as client_socket:
            client_socket.connect((KIOSK_HOST, KIOSK_PORT))
            client_socket.sendall(
                json.dumps(order_data, ensure_ascii=False).encode("utf-8")
            )
            response = client_socket.recv(1024).decode("utf-8")
            return f"주문 전송 성공: {response}"

    except Exception as e:
        error_msg = f"키오스크 전송 오류: {str(e)}"
        st.error(f"[ERROR] {error_msg}")
        return error_msg


def load_whisper_model():
    """Whisper 모델을 세션 상태에 초기화합니다."""
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = whisper.load_model(WHISPER_MODEL)
    return st.session_state.whisper_model


@tool
def speech_to_text(tool_input: str = "") -> str:
    """음성을 텍스트로 변환합니다."""
    try:
        st.info("말씀해주세요", icon="🎤")
        sd.default.samplerate = 16000
        sd.default.channels = 1
        recording = sd.rec(int(RECORD_SECONDS * 16000))
        sd.wait()
        wav_path = "temp_whisper.wav"
        sf.write(wav_path, recording, 16000)
        st.audio(wav_path, format="audio/wav")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = st.session_state.whisper_model
        result = model.transcribe(
            wav_path,
            language="ko",
            fp16=True if device == "cuda" else False,
            temperature=0.1,
            best_of=1,
            beam_size=1,
        )
        text = result.get("text", "").strip()
        if not text:
            return "음성이 인식되지 않았습니다. 다시 시도해 주세요."
        return text
    except Exception as e:
        return f"음성 인식 중 오류 발생: {str(e)}"


def text_to_speech(text: str):
    """텍스트를 한국어 음성(mp3)으로 변환하여 재생합니다."""
    try:
        tts = gTTS(text=text, lang="ko")
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"TTS 오류: {str(e)}")


# ===================== Whisper 모델 초기화 =====================


# ===================== 에이전트 초기화 =====================
def initialize_agent(tools: list):
    """LangChain 에이전트 및 프롬프트 초기화"""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )
    # serboway_prompt.txt 파일을 읽어서 system 프롬프트로 사용
    with open("serboway_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return create_tool_calling_agent(llm, tools, prompt)


# ===================== 메인 앱 (voice_agent UI 포맷) =====================
def main():
    st.set_page_config(page_title="서보웨이 AI 주문", page_icon="🥪")
    st.image("image/Menu.png")

    tools = [
        get_menu_list,
        get_sauce_list,
        get_vegetable_list,
        get_cheese_list,
        update_order,
        add_more_items,
        confirm_order,
        get_order_summary,
        send_order_to_kiosk,
        speech_to_text,
    ]

    if "initialized" not in st.session_state:
        initialize_session()
        st.session_state.initialized = True
        st.session_state.order_state = OrderState()
        load_whisper_model()
        st.session_state.agent = initialize_agent(tools)
        st.session_state.agent_executor = AgentExecutor(
            agent=st.session_state.agent, tools=tools, verbose=True, checkpointer=memory
        )
        # 세션 상태 및 에이전트 초기화
    if "messages" not in st.session_state:  # ← 들여쓰기 수정 (공백 4개)
        # 메뉴 목록을 실제로 가져와서 초기 메시지에 포함
        menu_list = get_menu_list("")
        initial_message = f"안녕하세요! Serboway입니다.\n\n{menu_list}\n\n어떤 메뉴를 주문하시겠습니까?"
        st.session_state.messages = [AIMessage(content=initial_message)]

    # 채팅 메시지 표시 (voice_agent 포맷) — TTS는 새로 추가된 마지막 AI 메시지에만 적용
    if "last_spoken_idx" not in st.session_state:
        st.session_state.last_spoken_idx = -1

    last_ai_idx = None
    for i, msg in enumerate(st.session_state.messages):
        if isinstance(msg, AIMessage):
            last_ai_idx = i

    for i, msg in enumerate(st.session_state.messages):
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                if i == last_ai_idx and i != st.session_state.last_spoken_idx:
                    text_to_speech(msg.content)
                    st.session_state.last_spoken_idx = i
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)

    # ===== 채팅 입력란 바로 아래에 음성 버튼 배치 =====
    user_input = st.chat_input("주문을 입력하세요 (텍스트 또는 음성 버튼 사용)")
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🎤 음성으로 주문하기", key="voice_btn_bottom"):
            with st.spinner("🎤 5초간 말씀해주세요..."):
                voice_result = speech_to_text.invoke({"tool_input": ""})
                if voice_result and not voice_result.startswith("음성 인식 오류"):
                    user_input = voice_result  # 음성 결과를 user_input에 저장

    # 텍스트 또는 음성 입력 처리
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("처리 중..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"input": user_input, "chat_history": st.session_state.messages}
                )
                answer = (
                    response["output"] if isinstance(response, dict) else str(response)
                )
                st.session_state.messages.append(AIMessage(content=answer))
                st.rerun()
            except Exception as e:
                st.error(f"처리 오류: {str(e)}")
                st.rerun()


if __name__ == "__main__":
    main()
