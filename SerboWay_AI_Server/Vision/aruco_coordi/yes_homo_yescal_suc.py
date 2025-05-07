import cv2
import cv2.aruco as aruco
import numpy as np
import os
import pickle

# ===== 카메라 보정값 =====
camera_matrix = np.array([[389.7681364, 0, 335.78433294],
                          [0, 388.59415835, 250.30158978],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([[-0.0995602, -0.0231152, -0.00138235, 0.00050955, 0.01000293]])

# ===== 기준 마커 월드 좌표 (단위: cm) =====
marker_world = {
    0: (0, 0,),
    1: (200, 0),
    2: (200, 100),
    3: (0, 100),
}

# ===== 호모그래피 계산 함수 =====
def build_homography(corners, ids):
    img_pts, world_pts = [], []
    if ids is None:
        return None

    for i, mid in enumerate(ids.flatten()):
        if mid in marker_world:
            c = corners[i][0]
            cx, cy = c[:, 0].mean(), c[:, 1].mean()
            img_pts.append([cx, cy])
            world_pts.append(marker_world[mid])

    if len(img_pts) < 4:
        return None

    img_pts = np.array(img_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)

    H, _ = cv2.findHomography(img_pts, world_pts)
    return H

# ===== 픽셀 → 월드 좌표 변환 =====
def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    world = H @ pt
    world /= world[2]
    return world[0], world[1]

# ===== 메인 루프 =====
def main(calibration_data):
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    params = aruco.DetectorParameters()
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']

    cached_H = None  # H 캐시

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y + h, x:x + w]

        corrected = cv2.resize(dst, (640, 480))
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

        # 기준 마커 있을 경우에만 H 갱신
        new_H = build_homography(corners, ids)
        if new_H is not None:
            cached_H = new_H

        # 마커가 인식되고, H가 유효할 때만 계산
        if ids is not None and cached_H is not None:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                px, py = c[:, 0].mean(), c[:, 1].mean()
                wx, wy = pixel_to_world(cached_H, px, py)

                # 시각화
                cv2.circle(corrected, (int(px), int(py)), 5, (0, 255, 0), -1)
                cv2.putText(corrected, f"ID:{mid}", (int(px) - 20, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # r
                # cv2.putText(corrected, f"({wx:.1f},{wy:.1f})", (int(px) - 30, int(py) + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(corrected, f"({int(round(wx))},{int(round(wy))})", (int(px) - 30, int(py) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(corrected, "No H matrix (need markers 0,1,3,4)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # aruco.drawDetectedMarkers(corrected, corners, ids)
        aruco.drawDetectedMarkers(corrected, corners)
        cv2.imshow("win", corrected)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===== 실행부 =====
if __name__ == "__main__":
    pkl_path = r'/home/addinedu/roscamp-repo-3/SerboWay_AI_Server/Vision/calibration/camera_calibration.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print('There is no any pkl file.')
        exit(1)

    main(calibration_data)
