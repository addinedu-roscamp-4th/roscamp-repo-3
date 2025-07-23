import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from pinky_lcd import LCD
from PIL import Image, ImageSequence
import threading
import time
class DriverLCDNode(Node):
    def __init__(self):
        super().__init__('DriverLCDNode')
        self.lcd = LCD()
        self.gif_files = {
            "left": Image.open("left.gif"),
            "right": Image.open("right.gif"),
            "forward": Image.open("forward.gif")
        }
        self.state = None
        self.gif_thread = None
        self.lock = threading.Lock()
        # cmd_vel 구독 (기존)
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
    def cmd_vel_callback(self, msg):
        # 기존 cmd_vel 콜백 로직
        if msg.angular.z > 0:
            new_state = "left"
        elif msg.angular.z < 0:
            new_state = "right"
        elif msg.linear.x > 0 and msg.angular.z == 0:
            new_state = "forward"
        else:
            new_state = None
        with self.lock:
            if new_state != self.state:
                self.state = new_state
                if self.gif_thread is None or not self.gif_thread.is_alive():
                    self.gif_thread = threading.Thread(target=self.show_gif)
                    self.gif_thread.start()
    def show_gif(self):
        while True:
            with self.lock:
                current_state = self.state
            if current_state is None:
                time.sleep(0.5)
                continue
            gif = self.gif_files[current_state]
            gif.seek(0)
            for frame in ImageSequence.Iterator(gif):
                with self.lock:
                    if self.state != current_state or self.state is None:
                        break
                self.lcd.img_show(frame)
                time.sleep(0.5)
            with self.lock:
                if self.state != current_state:
                    continue
    def destroy_node(self):
        self.lcd.close()
        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    node = DriverLCDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()