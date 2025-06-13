#!/usr/bin/env python3
# pip3 install pyserial
import serial
import time
import re
# 시리얼 포트 설정
#  - USB-to-Serial 케이블 사용 시: '/dev/ttyUSB0'
#  - GPIO UART 사용 시:    '/dev/ttyAMA0' 또는 '/dev/serial0'
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
TIMEOUT  = 1  # 읽기 타임아웃 (초)
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        # 포트가 열릴 때까지 잠깐 대기
        time.sleep(2)
        print(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE}bps")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return
    try:
        while True:
            # 한 줄 읽기 (줄끝 \r\n 포함)
            raw = ser.readline()
            if not raw:
                # 타임아웃 시 빈 바이트열 반환 → 반복
                continue
            # 디코딩 및 개행(\r\n) 제거
            line = raw.decode('utf-8', errors='replace').strip()
            # line에서 정수형만 추출
            match = re.match(r'^(\d+)$', line)
            if match:
                prox = int(match.group(1))
                if prox < 800:
                    print("No plate") # 근접센서값 미인식
                elif prox > 800:
                    print("Plate is placed") # 근접센서값 인식
            else:
                print(f"Ignored invalid data: {line}")
    except KeyboardInterrupt:
        print("\nStopping serial read.")
    finally:
        ser.close()
        print("Serial port closed.")
if __name__ == "__main__":
    main()