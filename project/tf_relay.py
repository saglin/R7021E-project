import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from tf2_msgs.msg import TFMessage


class TfRelay(Node):

    def __init__(self):
        super().__init__('tf_relay')

        # Dynamic TF: /tf
        dynamic_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Static TF: /tf_static
        # Must be reliable + transient local.
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.tf_pub = self.create_publisher(
            TFMessage,
            '/tf',
            dynamic_qos,
        )

        self.tf_sub = self.create_subscription(
            TFMessage,
            '/a200_0000/tf',
            self.tf_callback,
            dynamic_qos,
        )

        self.tf_static_pub = self.create_publisher(
            TFMessage,
            '/tf_static',
            static_qos,
        )

        self.tf_static_sub = self.create_subscription(
            TFMessage,
            '/a200_0000/tf_static',
            self.tf_static_callback,
            static_qos,
        )

        self.get_logger().info(
            'TF relay started: '
            '/a200_0000/tf -> /tf'
        )

        self.get_logger().info(
            'Static TF relay started: '
            '/a200_0000/tf_static -> /tf_static'
        )

    def tf_callback(self, msg):
        self.tf_pub.publish(msg)

    def tf_static_callback(self, msg):
        self.tf_static_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = TfRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
