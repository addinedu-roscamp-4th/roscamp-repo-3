#!/usr/bin/env python3
import math
import pickle
import cv2, cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node        import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg      import Empty
import heapq
from rclpy.duration import Duration

# Calibration and map settings
CALIB_PATH     = '/home/addinedu/camera_calibration.pkl'
MAP_WIDTH_CM   = 200.0
MAP_HEIGHT_CM  = 100.0
MARKER_SIZE_CM = 10.0
half = MARKER_SIZE_CM / 2.0
PPM = 5

# World marker coordinates (cm) for homography
marker_world = {
    0: (half,               half),
    1: (MAP_WIDTH_CM-half,  half),
    3: (half,               MAP_HEIGHT_CM-half),
    4: (MAP_WIDTH_CM-half,  MAP_HEIGHT_CM-half),
}

# ID→coord for bird’s-eye plotting & graph
id_to_coord = {
    # original nodes
    11: (44, 31),   # P1
    12: (138, 31),  # P2
    13: (144, 67),  # P3
    14: (42, 71),   # P4
    21: (70, 67),   # T1
    22: (94, 67),   # T2
    23: (118, 67),  # T3
    31: (144, 29),  # Pick
    32: (32,  67),  # Drop
    99: (92,  15),  # Home

    # 9 points between P1 (11) and P2 (12)
    101: ( 53.4, 31.0), 102: ( 62.8, 31.0), 103: ( 72.2, 31.0),
    104: ( 81.6, 31.0), 105: ( 91.0, 31.0), 106: (100.4, 31.0),
    107: (109.8, 31.0), 108: (119.2, 31.0), 109: (128.6, 31.0),

    # 9 points between P3 (13) and P4 (14)
    110: (133.8, 67.4), 111: (123.6, 67.8), 112: (113.4, 68.2),
    113: (103.2, 68.6), 114: ( 93.0, 69.0), 115: ( 82.8, 69.4),
    116: ( 72.6, 69.8), 117: ( 62.4, 70.2), 118: ( 52.2, 70.6),

    # 4 points between P2 (12) and P3 (13)
    119: (139.2, 38.2), 120: (140.4, 45.4),
    121: (141.6, 52.6), 122: (142.8, 59.8),

    # 4 points between P1 (11) and P4 (14)
    123: ( 43.6, 39.0), 124: ( 43.2, 47.0),
    125: ( 42.8, 55.0), 126: ( 42.4, 63.0),
}

def build_graph():
    node_coords = {
        'P1':       id_to_coord[11],
        'P2':       id_to_coord[12],
        'P3':       id_to_coord[13],
        'P4':       id_to_coord[14],
        'T1':       id_to_coord[21],
        'T2':       id_to_coord[22],
        'T3':       id_to_coord[23],
        'Pick':     id_to_coord[31],
        'Drop':     id_to_coord[32],
        'Home':     id_to_coord[99],
        
        # intermediates P1↔P2
        'P1_P2_1':  id_to_coord[101],
        'P1_P2_2':  id_to_coord[102],
        'P1_P2_3':  id_to_coord[103],
        'P1_P2_4':  id_to_coord[104],
        'P1_P2_5':  id_to_coord[105],
        'P1_P2_6':  id_to_coord[106],
        'P1_P2_7':  id_to_coord[107],
        'P1_P2_8':  id_to_coord[108],
        'P1_P2_9':  id_to_coord[109],
        # intermediates P3↔P4
        'P3_P4_1':  id_to_coord[110],
        'P3_P4_2':  id_to_coord[111],
        'P3_P4_3':  id_to_coord[112],
        'P3_P4_4':  id_to_coord[113],
        'P3_P4_5':  id_to_coord[114],
        'P3_P4_6':  id_to_coord[115],
        'P3_P4_7':  id_to_coord[116],
        'P3_P4_8':  id_to_coord[117],
        'P3_P4_9':  id_to_coord[118],
        # intermediates P2↔P3
        'P2_P3_1':  id_to_coord[119],
        'P2_P3_2':  id_to_coord[120],
        'P2_P3_3':  id_to_coord[121],
        'P2_P3_4':  id_to_coord[122],
        # intermediates P1↔P4
        'P1_P4_1':  id_to_coord[123],
        'P1_P4_2':  id_to_coord[124],
        'P1_P4_3':  id_to_coord[125],
        'P1_P4_4':  id_to_coord[126],
    }
    adj = {
        # original corners
        'P1':       ['P1_P2_1', 'P1_P4_1'],
        'P2':       ['P1_P2_9', 'P2_P3_1', 'Pick'],
        'P3':       ['P2_P3_4', 'P3_P4_1'],
        'P4':       ['P3_P4_9', 'P1_P4_4', 'Drop'],
        'T1': ['P3_P4_7'],  
        'T2': ['P3_P4_5'],  
        'T3': ['P3_P4_3'],  
        'Pick':     ['P2'],
        'Drop':     ['P4'],
        'Home':     ['P1_P2_5'],

        # P1↔P2 intermediates
        'P1_P2_1': ['P1', 'P1_P2_2'],
        'P1_P2_2': ['P1_P2_1', 'P1_P2_3'],
        'P1_P2_3': ['P1_P2_2', 'P1_P2_4'],
        'P1_P2_4': ['P1_P2_3', 'P1_P2_5'],
        'P1_P2_5': ['P1_P2_4', 'P1_P2_6'],
        'P1_P2_6': ['P1_P2_5', 'P1_P2_7'],
        'P1_P2_7': ['P1_P2_6', 'P1_P2_8'],
        'P1_P2_8': ['P1_P2_7', 'P1_P2_9'],
        'P1_P2_9': ['P1_P2_8', 'P2'],

        # P3↔P4 intermediates
        'P3_P4_1': ['P3',     'P3_P4_2'],
        'P3_P4_2': ['P3_P4_1','P3_P4_3'],
        'P3_P4_3': ['P3_P4_2','P3_P4_4'],
        'P3_P4_4': ['P3_P4_3','P3_P4_5'],
        'P3_P4_5': ['P3_P4_4','P3_P4_6'],
        'P3_P4_6': ['P3_P4_5','P3_P4_7'],
        'P3_P4_7': ['P3_P4_6','P3_P4_8'],
        'P3_P4_8': ['P3_P4_7','P3_P4_9'],
        'P3_P4_9': ['P3_P4_8','P4'],

        # P2↔P3 intermediates
        'P2_P3_1': ['P2',     'P2_P3_2'],
        'P2_P3_2': ['P2_P3_1','P2_P3_3'],
        'P2_P3_3': ['P2_P3_2','P2_P3_4'],
        'P2_P3_4': ['P2_P3_3','P3'],

        # P1↔P4 intermediates
        'P1_P4_1': ['P1',     'P1_P4_2'],
        'P1_P4_2': ['P1_P4_1','P1_P4_3'],
        'P1_P4_3': ['P1_P4_2','P1_P4_4'],
        'P1_P4_4': ['P1_P4_3','P4'],
    }
    # make bidirectional
    for u,nbrs in list(adj.items()):
        for v in nbrs:
            adj.setdefault(v,[])
            if u not in adj[v]:
                adj[v].append(u)
    name_to_id = {
        'P1':11,'P2':12,'P3':13,'P4':14,
        'T1':21,'T2':22,'T3':23,
        'Pick':31,'Drop':32,
        'Home':99,
        
    }
    id_to_name = {v:k for k,v in name_to_id.items()}
    return node_coords, adj, name_to_id, id_to_name

def heuristic(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def astar_graph(node_coords, adj, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(node_coords[start], node_coords[goal]), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    while open_set:
        f, cost, current, parent = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from[parent]
            return path[::-1]
        if current in came_from:
            continue
        came_from[current] = parent
        for neigh in adj[current]:
            d = heuristic(node_coords[current], node_coords[neigh])
            tg = cost + d
            if tg < gscore.get(neigh, float('inf')):
                gscore[neigh] = tg
                heapq.heappush(open_set, (tg + heuristic(node_coords[neigh], node_coords[goal]), tg, neigh, current))
    return []

class ArucoNavCommander(Node):
    def __init__(self):
        super().__init__('aruco_nav_commander')
        # 1초 뒤 좌표 발행 예약용 변수
        self.next_pub_time = None
        # Publishers
        self.pose_pub    = self.create_publisher(PointStamped, '/robot_pose',  10)
        self.front_pub   = self.create_publisher(PointStamped, '/robot_front', 10)
        self.target_pub  = self.create_publisher(PointStamped, '/target_point',10)
        self.arrived_pub = self.create_publisher(Empty,        '/arrived',     10)

        # Waypoint(6,7) publish state
        self.wp_sent   = set()
        self.wp_pubs   = { wid: self.create_publisher(PointStamped, f'/waypoint_{wid}', 10)
                           for wid in (6, 7) }
        self.wp_coords = {}
        # 선분 시작점 저장용
        self.start_pose = None

        # Load camera calibration
        with open(CALIB_PATH,'rb') as f:
            calib = pickle.load(f)
        self.mtx, self.dist    = calib['camera_matrix'], calib['dist_coeffs']
        self.aruco_dict        = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        self.params            = aruco.DetectorParameters()
        self.marker_length     = MARKER_SIZE_CM
        self.cap               = cv2.VideoCapture(2, cv2.CAP_V4L2)

        # Graph setup
        self.node_coords, self.adj, self.name_to_id, self.id_to_name = build_graph()
        self.current_pose  = None
        self.path_queue    = []
        self.active_target = None
        self.cached_H      = None

        # Key→ID mapping
        self.key_to_dest_id = {1:21, 2:22, 3:23, 8:31, 9:32, 0:99}

        # Visualization windows
        cv2.namedWindow("win",    cv2.WINDOW_NORMAL)
        cv2.namedWindow("tf_map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("win",    640, 480)
        cv2.resizeWindow("tf_map", int(MAP_WIDTH_CM*PPM), int(MAP_HEIGHT_CM*PPM))

        # Timers
        self.create_timer(1/30.0, self.vision_cb)
        #self.create_timer(0.5,    self.check_and_publish_next)

        # 안내 메시지
        self.get_logger().info("0:Home, 1:T1, 2:T2, 3:T3, 8:Pick, 9:Drop 키 입력 → A* 경로 생성 및 전송")

    # Utility: calculate path length
    def path_length(self, path):
        total = 0.0
        for u, v in zip(path, path[1:]):
            x1, y1 = self.node_coords[u]
            x2, y2 = self.node_coords[v]
            total += math.hypot(x2-x1, y2-y1)
        return total

    # Utility: enumerate all simple paths
    def dfs_all_paths(self, src, dst, visited=None):
        if visited is None:
            visited = [src]
        if src == dst:
            yield visited.copy()
        for nbr in self.adj[src]:
            if nbr in visited:
                continue
            visited.append(nbr)
            yield from self.dfs_all_paths(nbr, dst, visited)
            visited.pop()

    def build_homography(self, corners, ids):
        if ids is None: return None
        img_pts, world_pts = [], []
        for i, mid in enumerate(ids.flatten()):
            if mid in marker_world:
                pts = corners[i][0]
                img_pts.append([float(pts[:,0].mean()), float(pts[:,1].mean())])
                world_pts.append(marker_world[mid])
        if len(img_pts) < 4: return None
        H, _ = cv2.findHomography(np.array(img_pts,np.float32), np.array(world_pts,np.float32))
        return H

    def pixel_to_world(self, H, px, py):
        pt = np.array([px,py,1.0],dtype=np.float32)
        w  = H @ pt; w /= w[2]
        return float(w[0]), float(w[1])

    def vision_cb(self):
        now = self.get_clock().now()
        # 카메라 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            cv2.waitKey(1)
            return

        # Grayscale & detect ArUco markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params
        )
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Homography 갱신
        Hn = self.build_homography(corners, ids)
        if Hn is not None:
            self.cached_H = Hn

        wx_r = wy_r = wx_f = wy_f = None

        # 로봇 pose & front publish + win 창에 화살표 그리기
        if ids is not None and self.cached_H is not None and 5 in ids.flatten():
            # Pose 계산
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.mtx, self.dist
            )
            i5 = list(ids.flatten()).index(5)
            R, _ = cv2.Rodrigues(rvecs[i5][0])
            c5 = corners[i5][0]
            px, py = float(c5[:,0].mean()), float(c5[:,1].mean())
            wx_r, wy_r = self.pixel_to_world(self.cached_H, px, py)
            self.current_pose = (wx_r, wy_r)

            # front point 계산
            front3d = np.array([[0.0, self.marker_length, 0.0]], dtype=np.float32)
            camf = (R @ front3d.T + tvecs[i5][0].reshape(3,1)).T
            imgpts, _ = cv2.projectPoints(
                camf, np.zeros(3), np.zeros(3), self.mtx, self.dist
            )
            fx, fy = imgpts[0].ravel().astype(int)
            wx_f, wy_f = self.pixel_to_world(self.cached_H, fx, fy)

            # publish robot pose
            p = PointStamped()
            p.header.frame_id = 'map'
            p.header.stamp    = self.get_clock().now().to_msg()
            p.point.x, p.point.y, p.point.z = wx_r, wy_r, 0.0
            self.pose_pub.publish(p)

            # publish robot front
            f = PointStamped()
            f.header = p.header
            f.point.x, f.point.y, f.point.z = wx_f, wy_f, 0.0
            self.front_pub.publish(f)
            # win 창에 전방 화살표 그리기
            cv2.arrowedLine(
                frame,
                (int(px), int(py)),
                (int(fx), int(fy)),
                (0, 255, 0),
                2,
                tipLength=0.2
            )
                      
        # ─────────── 도착 판정 로직 ───────────
        # wx_r/wy_r이 None이 아닐 때만 실행
        if self.active_target and self.current_pose is not None:
            dx = self.active_target[0] - wx_r
            dy = self.active_target[1] - wy_r
            if math.hypot(dx, dy) <= 2.0:
                 # 도착 신호 발행
                 self.arrived_pub.publish(Empty())
        # ────────────────────────────────────────────────────────────
        # ─────────── 예약된 다음 좌표 발행 체크 ───────────
        if self.next_pub_time is not None and now >= self.next_pub_time:
            # 실제 발행
            self.check_and_publish_next()
            # 다시 발행 안 되도록 초기화
            self.next_pub_time = None

        # waypoint (6,7) publish once
        if ids is not None and self.cached_H is not None:
            for wid in (6, 7):
                if wid in ids.flatten() and wid not in self.wp_sent:
                    idx = list(ids.flatten()).index(wid)
                    cw = corners[idx][0]
                    pxw, pyw = float(cw[:,0].mean()), float(cw[:,1].mean())
                    wx, wy = self.pixel_to_world(self.cached_H, pxw, pyw)
                    msg = PointStamped()
                    msg.header.frame_id = 'map'
                    msg.header.stamp    = self.get_clock().now().to_msg()
                    msg.point.x, msg.point.y, msg.point.z = wx, wy, 0.0
                    self.wp_pubs[wid].publish(msg)
                    self.wp_coords[wid] = (wx, wy)
                    self.wp_sent.add(wid)

        # 원본 영상 표시
        cv2.imshow("win", frame)

        # Bird’s-eye view + path_queue, active_target, 선분 그리기
        if self.cached_H is not None:
            S = np.array([[PPM, 0, 0],
                          [0, -PPM, int(MAP_HEIGHT_CM*PPM)],
                          [0, 0, 1]], np.float32)
            tf_map = cv2.warpPerspective(
                frame,
                S @ self.cached_H,
                (int(MAP_WIDTH_CM*PPM), int(MAP_HEIGHT_CM*PPM))
            )

            # path_queue 표시
            for x, y in self.path_queue:
                cx, cy = int(x*PPM), int(int(MAP_HEIGHT_CM*PPM) - y*PPM)
                cv2.circle(tf_map, (cx, cy), 6, (0, 255, 0), -1)

            # active_target 표시
            if self.active_target:
                tx, ty = self.active_target
                cx, cy = int(tx*PPM), int(int(MAP_HEIGHT_CM*PPM) - ty*PPM)
                cv2.circle(tf_map, (cx, cy), 8, (0, 0, 255), -1)

            # 선분 그리기: start_pose → active_target
            if self.start_pose is not None and self.active_target:
                sx, sy = self.start_pose
                tx, ty = self.active_target
                x0, y0 = int(sx*PPM), int(int(MAP_HEIGHT_CM*PPM) - sy*PPM)
                x1, y1 = int(tx*PPM), int(int(MAP_HEIGHT_CM*PPM) - ty*PPM)
                cv2.line(tf_map, (x0, y0), (x1, y1), (255, 0, 0), 2)

            cv2.imshow("tf_map", tf_map)

        # 키 입력 처리: A* 경로 생성 → 필터링 → start_pose 고정 → 첫 좌표 퍼블리시
        key = cv2.waitKey(1) & 0xFF
        if chr(key).isdigit():
            pid = int(chr(key))
            if pid in self.key_to_dest_id and self.current_pose is not None:
                self.active_target = None

                # start/goal 설정
                valid = [n for n in self.node_coords if n not in ('T1','T2','T3')]
                start = min(
                    valid,
                    key=lambda n: math.hypot(
                        self.current_pose[0] - self.node_coords[n][0],
                        self.current_pose[1] - self.node_coords[n][1]
                    )
                )
                goal_name = self.id_to_name[self.key_to_dest_id[pid]]
                all_paths = list(self.dfs_all_paths(start, goal_name))
                if not all_paths:
                    self.get_logger().warn(f"⚠️ 경로 없음: {start} → {goal_name}")
                    return

                # 최단 경로 선택
                best_path, _ = min(
                    ((p, self.path_length(p)) for p in all_paths),
                    key=lambda x: x[1]
                )
                self.get_logger().info(f"✅ 후보 경로: {best_path}")

                # 필터링 로직
                corner_nodes = {'P1','P2','P3','P4'}
                key_nodes    = {'Home','Pick','Drop','T1','T2','T3'}
                if best_path[0] in key_nodes:
                    si = 1
                else:
                    si = 0
                filtered = [best_path[si]]
                for i in range(si + 1, len(best_path)):
                    node, prev = best_path[i], best_path[i - 1]
                    if node in corner_nodes:
                        filtered.append(node)
                    elif node in key_nodes:
                        if prev not in filtered:
                            filtered.append(prev)
                        filtered.append(node)
                # 경로 설정 & 즉시 퍼블리시 (start_pose 고정은 다음 함수로 이동)
                self.path_queue = [self.node_coords[n] for n in filtered]
                self.check_and_publish_next()

        elif key == ord('q'):
            rclpy.shutdown()

    def check_and_publish_next(self):
        if self.active_target or not self.path_queue:
            return

        # 발행 시점의 로봇 위치를 start_pose로 저장
        self.start_pose = self.current_pose
        
        x, y = self.path_queue.pop(0)
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), 0.0
        self.target_pub.publish(msg)
        self.active_target = (x, y)
        self.get_logger().info(f"📍 좌표 전송: ({x:.1f}, {y:.1f})")

def main():
    rclpy.init()
    node = ArucoNavCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
