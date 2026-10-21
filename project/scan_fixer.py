import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


'''
Takes the original scan from the lidar and replaces all no-return beams with the maximum range of the sensor

Subscription topics:
    /a200_0000/sensors/lidar2d_0/scan (sensor_msgs/LaserScan)
Publishing topics:
    /scan_fixed (sensor_msgs/LaserScan)

'''
class ScanFixer(Node):
    def __init__(self):
        super().__init__('scan_fixer')
        self.pub = self.create_publisher(
            LaserScan, '/scan_fixed', qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/a200_0000/sensors/lidar2d_0/scan',
            self.cb, qos_profile_sensor_data)

    def cb(self, msg: LaserScan):
        # Keep no-return beams at the sensor limit, beyond SLAM's hit threshold.
        msg.ranges = [
            msg.range_max if math.isinf(r) and r > 0 else r
            for r in msg.ranges
        ]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ScanFixer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()