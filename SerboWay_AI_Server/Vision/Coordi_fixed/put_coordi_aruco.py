import cv2
import cv2.aruco as aruco
import numpy as np

# ===== 카메라 보정값 (예시로 대체하세요) =====
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 왜곡 없는 경우

# ===== 마커 ID별 월드 좌표 (임의 설정) =====
marker_world_coords = {
    0: (0, 0),
    1: (200, 0),
    2: (200, 100),
    3: (0, 100),
}

# ===== 카메라 열기 =====
cap = cv2.VideoCapture(2)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = c.mean(axis=0).astype(int)
            cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)

            if marker_id in marker_world_coords:
                world_coord = marker_world_coords[marker_id]
                text_id = f"ID: {marker_id}"
                text_coord = f"{world_coord}"

                # 텍스트 크기 측정
                (text_w1, text_h1), _ = cv2.getTextSize(text_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                (text_w2, text_h2), _ = cv2.getTextSize(text_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # 텍스트 위치 계산 (가운데 정렬)
                id_pos = (center[0] - text_w1 // 2, center[1] - 10)
                coord_pos = (center[0] - text_w2 // 2, center[1] + 15)

                # 텍스트 출력
                cv2.putText(frame, text_id, id_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, text_coord, coord_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


    cv2.imshow("ArUco Marker Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
