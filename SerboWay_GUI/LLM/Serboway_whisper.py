import re
import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Union
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from gtts import gTTS
import io
import whisper
import sounddevice as sd
import soundfile as sf

# ===== 상태 관리를 위한 클래스 정의 =====
class OrderState(TypedDict):
    messages: Annotated[List[Union[AIMessage, HumanMessage]], add_messages]  # 메시지 리스트 관리
    order: dict  # 주문 정보 저장

# ===== Whisper 모델을 이용한 음성 인식 함수 =====
def speech_to_text():
    st.info("말씀해주세요...", icon="🎤")
    sd.default.samplerate = 16000
    sd.default.channels = 1
    recording = sd.rec(int(5 * 16000))  # 5초 동안 음성 녹음
    sd.wait()
    wav_path = "temp_whisper.wav"
    sf.write(wav_path, recording, 16000)
    model = whisper.load_model("base")
    result = model.transcribe(wav_path, language="ko")  # 음성을 한국어 텍스트로 변환
    return result["text"]

# ===== 주문 텍스트를 파싱하여 주문 정보를 추출하는 함수 =====
def parse_order(text: str) -> dict:
    menu_match = re.search(r"(햄|불고기|새우)", text)
    menu_map = {
        "햄": "햄 샌드위치",
        "불고기": "불고기 샌드위치",
        "새우": "새우 샌드위치"
    }
    menu = menu_map[menu_match.group(0)] if menu_match else None

    sauces = re.findall(r"(이탈리안|칠리)", text)
    cheeses = re.findall(r"(슈레드 치즈|모짜렐라 치즈|슬라이스 치즈)", text)
    vegetables = re.findall(r"(로메인|바질|양상추)", text)
    etc = re.findall(r"(베이컨)", text)
    done = bool(re.search(r"(완료|그만|끝|주문\s*완료)", text))

    return {
        "menu": menu,
        "sauce": list(set(sauces)),
        "cheese": list(set(cheeses)),
        "vegetable": list(set(vegetables)),
        "etc": list(set(etc)),
        "done": done,
        "confirmed": False,
    }

# ===== 주문 내용을 보기 좋게 포맷하는 함수 =====
def format_order_summary(order: dict) -> str:
    parts = [f"메뉴: {order['menu']}"] if order.get("menu") else []
    if order["sauce"]:
        parts.append(f"소스: {', '.join(order['sauce'])}")
    if order["cheese"]:
        parts.append(f"치즈: {', '.join(order['cheese'])}")
    if order["vegetable"]:
        parts.append(f"야채: {', '.join(order['vegetable'])}")
    if order["etc"]:
        parts.append(f"추가: {', '.join(order['etc'])}")
    return "\n".join(parts)

# ===== 주문을 처리하는 노드 =====
def process_order_node(state: OrderState):
    last_msg = state["messages"][-1].content
    order = parse_order(last_msg)

    # 메뉴가 선택되지 않은 경우
    if not order["menu"]:
        return {
            "messages": [AIMessage("메뉴를 선택해주세요. (햄 샌드위치 / 불고기 샌드위치 / 새우 샌드위치)")],
            "order": order
        }

    # 부가 재료가 없는 경우
    if not any([order["sauce"], order["cheese"], order["vegetable"], order["etc"]]):
        return {
            "messages": [AIMessage(f"{order['menu']}를 선택하셨습니다. 추가할 소스, 치즈, 야채 또는 베이컨이 있으신가요?")],
            "order": order
        }

    # 주문 완료 신호가 없는 경우
    if not order["done"]:
        summary = format_order_summary(order)
        return {
            "messages": [AIMessage(f"현재까지의 주문 내역입니다:\n{summary}\n주문을 완료하시려면 '완료'라고 말씀해주세요.")],
            "order": order
        }

    # 주문 완료 확인 메시지
    summary = format_order_summary(order)
    return {
        "messages": [AIMessage(f"{summary}\n로 주문하시겠습니까? (네/아니오)")],
        "order": order
    }

# ===== 주문 확인 처리 노드 =====
def confirm_order_node(state: OrderState):
    last_msg = state["messages"][-1].content.lower()
    if "네" in last_msg:
        state["order"]["confirmed"] = True
        summary = format_order_summary(state["order"])
        return {
            "messages": [AIMessage(f"✅ 주문 완료! 다음과 같이 준비하겠습니다:\n{summary}")],
            "order": state["order"]
        }
    # 주문을 다시 시작 요청
    return {
        "messages": [AIMessage("🔄 주문을 다시 시작해주세요.")],
        "order": {}
    }

# ===== 메시지를 분석하여 다음 노드를 결정하는 라우팅 함수 =====
def route_message(state: OrderState):
    last_msg = state["messages"][-1].content.lower()
    if state["order"].get("confirmed"):
        return END
    if any(kw in last_msg for kw in ["네", "아니오"]):
        return "confirm"
    return "process"

# ===== Streamlit을 이용한 웹 인터페이스 구성 =====
st.set_page_config(page_title="서보웨이 AI 주문", page_icon="🥪")
st.title("🥪 서보웨이 AI 주문 시스템")

# image 경로
image_url="Menu.png"

# 웹상의 이미지 표시
st.image(image_url)

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage("어서오세요! 주문하실 샌드위치를 선택해주세요. (햄/불고기/새우)")]

# 메시지 채팅 창에 출력
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# 텍스트 및 음성 입력
input_col, voice_col = st.columns([5, 1])
with input_col:
    text_input = st.chat_input("주문을 입력하세요...")
with voice_col:
    if st.button("🎤", use_container_width=True):
        text_input = speech_to_text()

# 워크플로우 처리 및 결과 출력
if text_input:
    st.session_state.messages.append(HumanMessage(text_input))
    workflow = StateGraph(OrderState)
    workflow.add_node("process", process_order_node)
    workflow.add_node("confirm", confirm_order_node)
    workflow.add_conditional_edges("process", route_message, {"confirm": "confirm", "process": END})
    workflow.add_edge("confirm", END)
    workflow.set_entry_point("process")
    compiled_workflow = workflow.compile()

    result = compiled_workflow.invoke({"messages": st.session_state.messages, "order": {}})
    ai_response = result["messages"][-1]
    st.session_state.messages.append(ai_response)
    st.chat_message("assistant").write(ai_response.content)

    # 추가된 조건: 빈 응답 체크
    if ai_response.content.strip():
        tts = gTTS(ai_response.content, lang="ko")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf.read(), format="audio/mp3")
    else:
        st.warning("응답 텍스트가 비어 있어 음성 출력을 건너뜁니다.")

