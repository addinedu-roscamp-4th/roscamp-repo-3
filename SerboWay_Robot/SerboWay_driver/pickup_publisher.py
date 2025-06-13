import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .driver_pickup import Pickup
from std_msgs.msg import Int32
import time
class PickupPublisher(Node):
    def __init__(self):
        super().__init__('pickup_publisher')
        self.pickup = Pickup()
        self.pickup_publisher = self.create_publisher(
            Int32,
            '/pinky_pickup',
            10
        )
        self.timer = self.create_timer(0.1, self.pickup_callback)
    def pickup_callback(self):
        self.pickup.ser.reset_input_buffer()
        #time.sleep(0.01)
        msg = Int32()
        value = self.pickup.read_prox()
        if value is None:
            return
        self.get_logger().info(f"read_prox() returned: {value}")
        msg.data = value
        self.pickup_publisher.publish(msg)
def main(args=None):
    rclpy.init(args=args)
    publisher = PickupPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()