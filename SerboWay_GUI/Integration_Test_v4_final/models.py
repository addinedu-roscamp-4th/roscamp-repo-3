# ===================== 공통 Pydantic 모델 =====================
# serbo_kiosk.py와 serbo_kiosk_voice.py에서 공유하는 주문 데이터 스키마
from pydantic import BaseModel
from typing import List


class MenuItemSchema(BaseModel):
    name: str
    price: int
    qty: int
    sauce: str
    vegetable: str
    cheese: str


class OrderDataSchema(BaseModel):
    menu: List[MenuItemSchema]
    table_number: int
    timestamp: str
