# ===================== SerboWay 공통 설정 =====================
# 환경변수(.env)에서 값을 읽고, 없으면 기본값 사용
import os
from dotenv import load_dotenv

load_dotenv()

# ============== 네트워크 설정 ==============
ORDER_SERVER_URL = os.getenv("ORDER_SERVER_URL", "http://127.0.0.1:5004")
KIOSK_HOST = os.getenv("KIOSK_HOST", "127.0.0.1")
KIOSK_PORT = int(os.getenv("KIOSK_PORT", "5050"))

# ============== Streamlit 설정 ==============
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_SCRIPT = os.getenv("STREAMLIT_SCRIPT", "serbo_kiosk_voice.py")

# ============== 테이블/주문 설정 ==============
TABLE_NUM = int(os.getenv("TABLE_NUM", "1"))
