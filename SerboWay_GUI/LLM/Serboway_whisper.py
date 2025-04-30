import re
import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Union
from langchain_core.messages import AIMessage, HumanMessage
import speech_recognition as sr
from langgraph.graph.message import add_messages
from gtts import gTTS
import io
import whisper
import sounddevice as sd
import soundfile as sf

# ===== 상태 관리 클래스 =====
class OrderState(TypedDict):
    messages: Annotated[List[Union[AIMessage, HumanMessage]], add_messages]
    order: dict

# ===== 음성 인식 모듈 =====
def speech_to_text():
    """Whisper 모델로 음성을 텍스트로 변환"""
    st.info("말씀해주세요...", icon="🎤")
    # 1) 녹음 설정 (5초, 16kHz, mono)
    sd.default.samplerate = 16000
    sd.default.channels = 1
    recording = sd.rec(int(5 * 16000))  # 5초간 녹음
    sd.wait()

    # 2) 임시 WAV 파일로 저장
    wav_path = "temp_whisper.wav"
    sf.write(wav_path, recording, 16000)

    # 3) Whisper 로드 & 추론
    model = whisper.load_model("base")      # "tiny", "small" 등으로 경량화 가능
    result = model.transcribe(wav_path, language="ko")
    return result["text"]

# ===== 주문 처리 엔진 =====
def parse_order(text: str) -> dict:
    """
    재료 주문 파싱  
    - 햄, 치즈, 양상추만 인식  
    - 중복 가능 
    - 최대 3가지까지
    - 사용자가 '완료', '끝', '그만' 등의 키워드를 말했는지 확인
    """
    # 1) 재료 키워드만 골라내고 순서 유지하면서 중복 제거
    raw = re.findall(r"(햄|치즈|양상추)", text)
    ingredients = list(dict.fromkeys(raw))[:3]

    # 2) 사용자가 완료 의사를 표현했는지 확인
    done = bool(re.search(r"(완료|끝|그만|주문\s*완료)", text))

    return {
        "ingredients": ingredients,
        "done": done,
        "confirmed": False,  # 기존 워크플로우 호환용
    }


def process_order_node(state: OrderState):
    """재료 주문 처리 노드"""
    last_msg = state["messages"][-1].content
    order = parse_order(last_msg)
    ingredients = order["ingredients"]
    done = order["done"]

    # 1) 아직 아무 재료도 선택하지 않은 경우
    if not ingredients:
        return {
            "messages": [
                AIMessage("재료를 선택해주세요 (햄/치즈/양상추), 최대 3개까지 고를 수 있어요.")
            ],
            "order": order,
        }

    # 2) 3개 미만 선택 & 사용자가 아직 '완료'를 안 말한 경우
    if not done and len(ingredients) < 3:
        sel = ", ".join(ingredients)
        return {
            "messages": [
                AIMessage(
                    f"지금까지 선택한 재료: {sel}.\n"
                    "추가할 재료가 있으면 말씀해주세요. "
                    "선택을 마치셨으면 '완료'라고 말해주세요."
                )
            ],
            "order": order,
        }

    # 3) 완료 키워드 입력 혹은 3개 다 채운 경우 → 확인 요청
    sel = ", ".join(ingredients)
    return {
        "messages": [
            AIMessage(f"{sel} 재료로 주문을 확정하시겠습니까? (네/아니오)")
        ],
        "order": order,
    }


def confirm_order_node(state: OrderState):
    """주문 확인 노드"""
    last_msg = state["messages"][-1].content.lower()
    if "네" in last_msg:
        state["order"]["confirmed"] = True
        return {
            "messages": [AIMessage("✅ 주문 완료! 매장에서 바로 준비합니다.")],
            "order": state["order"],
        }
    else:
        return {
            "messages": [AIMessage("🔄 주문을 다시 시작해주세요.")],
            "order": {},
        }

# ===== 대화 흐름 제어 =====
def route_message(state: OrderState):
    last_msg = state["messages"][-1].content.lower()
    if state["order"].get("confirmed"):
        return END
    if any(kw in last_msg for kw in ["네", "아니오"]):
        return "confirm"
    return "process"

# ===== Streamlit UI 설정 =====
st.set_page_config(page_title="서보웨이 AI 주문", page_icon="🥪")
st.title("🥪 서보웨이 AI 주문 시스템")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage("어서오세요! 어떤 메뉴를 주문하시겠어요?")
    ]

# 채팅 기록 표시
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# 입력 처리
input_col, voice_col = st.columns([5, 1])
with input_col:
    text_input = st.chat_input("주문을 입력하세요...")
with voice_col:
    if st.button("🎤", use_container_width=True):
        text_input = speech_to_text()

if text_input:
    # 사용자 입력 처리
    st.session_state.messages.append(HumanMessage(text_input))
    st.chat_message("user").write(text_input)

    # 주문 처리 그래프 설정
    workflow = StateGraph(OrderState)
    workflow.add_node("process", process_order_node)
    workflow.add_node("confirm", confirm_order_node)
    workflow.add_conditional_edges(
        "process",
        route_message,
        {"confirm": "confirm", "process": END},
    )
    workflow.add_edge("confirm", END)
    workflow.set_entry_point("process")
    compiled_workflow = workflow.compile()

    # 워크플로우 실행
    result = compiled_workflow.invoke(
        {"messages": st.session_state.messages, "order": {}}
    )

    # AI 응답 표시
    ai_response = result["messages"][-1]
    st.session_state.messages.append(ai_response)
    st.chat_message("assistant").write(ai_response.content)

    # TTS 변환 및 재생
    tts = gTTS(ai_response.content, lang="ko")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    st.audio(buf.read(), format="audio/mp3")

    # 재주문 처리
    if "다시 주문" in ai_response.content:
        st.session_state.messages = [
            AIMessage("새 주문을 시작합니다. 메뉴를 선택해주세요.")
        ]
        st.rerun()
