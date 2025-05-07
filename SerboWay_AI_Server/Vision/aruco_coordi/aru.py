import cv2
import cv2.aruco as aruco
import numpy as np
# —— 1) 카메라 보정값 (예시) ——
camera_matrix = np.array([[1000, 0,   640],
                          [0,    1000, 360],
                          [0,    0,     1]], dtype=np.float32)

dist_coeffs = np.zeros((5,1))

# —— 2) 월드 좌표계에서 “꼭짓점” 마커 ID 별 실제 좌표 (단위: cm 등) ——
marker_world = {
    0: (  0,   0),   # 왼쪽 아래
    1: (200,   0),   # 오른쪽 아래
    4: (200, 100),   # 오른쪽 위
    3: (  0, 100),   # 왼쪽 위
}

def build_homography(corners, ids):
    """
    4개의 꼭짓점 마커에 대해:
      image_pts = [ (u0,v0), (u1,v1), (u4,v4), (u3,v3) ]
      world_pts = [ (x0,y0), (x1,y1), (x4,y4), (x3,y3) ]
    로 매핑하는 호모그래피 H를 계산해서 반환.
    """
    img_pts = []
    world_pts = []
    for i, mid in enumerate(ids.flatten()):
        if mid in marker_world:
            # 마커의 4개 코너 중 가운데 점을 씁니다.
            c = corners[i][0]   # shape (4,2)
            cx, cy = c[:,0].mean(), c[:,1].mean()
            img_pts.append([cx, cy])
            world_pts.append(marker_world[mid])
    img_pts = np.array(img_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)
    if len(img_pts) < 4:
        return None
    H, _ = cv2.findHomography(img_pts, world_pts)
    return H
def pixel_to_world(H, px, py):
    """
    호모그래피 H를 이용해 (px,py,1) → (X,Y,1) 변환하고
    동차 좌표(normalized)로 리턴.
    """
    pt = np.array([px, py, 1.0])
    world = H.dot(pt)
    world /= world[2]
    return world[0], world[1]
def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("카메라 열기 실패"); return
    # ArUco 설정
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    params     = aruco.DetectorParameters()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None:
            cv2.imshow("win", frame)
            if cv2.waitKey(33) & 0xFF==ord('q'): break
            continue
        # 1) 4개 꼭짓점 마커로 호모그래피 구축
        H = build_homography(corners, ids)
        if H is None:
            cv2.putText(frame, "Need all 4 corner markers!", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        else:
            # 2) 전체 인식된 마커에 대해
            for i, mid in enumerate(ids.flatten()):
                # 픽셀 중심
                c = corners[i][0]
                px, py = c[:,0].mean(), c[:,1].mean()
                # world 좌표 계산
                wx, wy = pixel_to_world(H, px, py)
                # 화면에 그리기
                cv2.circle(frame, (int(px),int(py)), 5, (0,255,0), -1)
                cv2.putText(frame, f"ID:{mid}", (int(px)-20,int(py)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2)
                cv2.putText(frame, f"({wx:.1f},{wy:.1f})", (int(px)-30,int(py)+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
                # 5번 마커에 대해서만 월드 좌표 출력
                if mid == 5:
                    print(f"Marker 5 world coord = ({wx:.2f}, {wy:.2f})")
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("win", frame)
        if cv2.waitKey(33) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()