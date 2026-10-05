import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        camera_image_topic = '/a200_0000/sensors/camera_0/color/image'
        YOLO_MODEL_PATH = 'yolov8n.pt'

        self.model = YOLO(YOLO_MODEL_PATH)

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(Image, camera_image_topic, 
                                                     self.image_callback, 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.track(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("RealSense", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()