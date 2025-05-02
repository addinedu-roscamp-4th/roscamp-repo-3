import cv2
import cv2.aruco as aruco
import numpy as np
import os
import pickle

#마커 사이즈 측정해서 넣어줘야함

# # —— 1) 카메라 보정값 (예시) ——
# camera_matrix = np.array([[1000, 0,   640],
#                           [0,    1000, 360],
#                           [0,    0,     1]], dtype=np.float32)

camera_matrix = np.array([[389.7681364, 0,   335.78433294],
                          [0,    388.59415835, 250.30158978],
                          [0,    0,     1]], dtype=np.float32)

"""
 [[389.7681364    0.         335.78433294]

 [  0.         388.59415835 250.30158978]

 [  0.           0.           1.        ]]
"""

# dist_coeffs = np.zeros((5,1))
dist_coeffs = np.array([[-0.0995602,  -0.0231152,  -0.00138235,  0.00050955,  0.01000293]])

# —— 2) 월드 좌표계에서 “꼭짓점” 마커 ID 별 실제 좌표 (단위: cm 등) ——
marker_world = {
    0: (  0,   0),   # 왼쪽 아래
    1: (200,   0),   # 오른쪽 아래
    4: (200, 100),   # 오른쪽 위
    3: (  0, 100),   # 왼쪽 위
}

def coordi_fixed(corners, ids, corrected):
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            center = c.mean(axis=0).astype(int)
            cv2.circle(corrected, tuple(center), 5, (0, 255, 0), -1)

            if marker_id in marker_world:
                world_coord = marker_world[marker_id]
                text_id = f"ID: {marker_id}"
                text_coord = f"{world_coord}"

                # 텍스트 크기 측정
                (text_w1, text_h1), _ = cv2.getTextSize(text_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                (text_w2, text_h2), _ = cv2.getTextSize(text_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # 텍스트 위치 계산 (가운데 정렬)
        id_pos = (center[0] - text_w1 // 2, center[1] - 10)
        coord_pos = (center[0] - text_w2 // 2, center[1] + 15)

        # 텍스트 출력
        cv2.putText(corrected, text_id, id_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(corrected, text_coord, coord_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    else:
        cv2.putText(corrected, "Need all 4 corner markers!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

def pixel_to_world(ids, px, py):
    """
    호모그래피 H를 이용해 (px,py,1) → (X,Y,1) 변환하고
    동차 좌표(normalized)로 리턴.
    """
    
    pt = np.array([px, py, 1.0])
    world = ids.dot(pt)
    world /= world[2]
    return world[0], world[1]

def main(calibration_data):
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("카메라 열기 실패"); return
    
    # ArUco 설정
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    params     = aruco.DetectorParameters()

    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        corrected = cv2.resize(dst, (640, 480))
    
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    
        # if ids is None:
        #     cv2.imshow("win", frame)
        #     if cv2.waitKey(33) & 0xFF==ord('q'): break
        #     continue
        
        for i, mid in enumerate(ids.flatten()):
            # 픽셀 중심
            c = corners[i][0]
            px, py = c[:,0].mean(), c[:,1].mean()
            # world 좌표 계산
            wx, wy = pixel_to_world(ids, px, py)
            # 화면에 그리기
            cv2.circle(corrected, (int(px),int(py)), 5, (0,255,0), -1)
            cv2.putText(corrected, f"ID:{mid}", (int(px)-20,int(py)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2)
            cv2.putText(corrected, f"({wx:.1f},{wy:.1f})", (int(px)-30,int(py)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
            
            # # 5번 마커에 대해서만 월드 좌표 출력
            # if mid == 5:
            #     print(f"Marker 5 world coord = ({wx:.2f}, {wy:.2f})")
    
        aruco.drawDetectedMarkers(corrected, corners, ids)
        
        cv2.imshow("win", corrected)
        if cv2.waitKey(33) & 0xFF==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return corners, ids, corrected

if __name__ == "__main__":
    pkl_path = r'/home/addinedu/roscamp-repo-3/SerboWay_AI_Server/Vision/calibration/camera_calibration.pkl'
    if os.path.exists(pkl_path):
        # print("Loading existing calibration data...")
        with open(pkl_path, 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print('There is no any pkl file.')
    
    main(calibration_data)