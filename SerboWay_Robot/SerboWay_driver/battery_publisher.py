import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .battery import Battery
from std_msgs.msg import Float32

class BatteryPublisher(Node):
    def __init__(self):
        super().__init__('battery_publihser')
        self.battery = Battery()

        self.battery_publisher = self.create_publisher(
            Float32,
            '/pinky_battery_present',
            10
        )
        self.timer = self.create_timer(0.05, self.battery_callback)

    def battery_callback(self):
        msg = Float32()
        msg.data = self.battery.get_battery()

        self.battery_publisher.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)
    publisher = BatteryPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
