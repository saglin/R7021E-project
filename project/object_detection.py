import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
from ultralytics import YOLO
import numpy as np

from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point

class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        camera_image_topic = '/a200_0000/sensors/camera_0/color/image'
        camera_image_info_topic = '/a200_0000/sensors/camera_0/color/camera_info'
        camera_depth_topic = '/a200_0000/sensors/camera_0/depth/image'
        camera_depth_info_topic = '/a200_0000/sensors/camera_0/depth/camera_info'
        YOLO_MODEL_PATH = 'yolov8n.pt'
        self.target_frame = 'base_link'

        self.model = YOLO(YOLO_MODEL_PATH)

        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(Image, camera_image_topic, 
                                                     self.image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, camera_depth_topic,
                                                           self.depth_callback, 10)
        self.image_info_subscription = self.create_subscription(CameraInfo, camera_image_info_topic,
                                                                self.image_info_callback, 10)
        self.depth_info_subscription = self.create_subscription(CameraInfo, camera_depth_info_topic,
                                                                self.depth_info_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_model = PinholeCameraModel()

        self.depth_frame = None
        self.camera_info = None
        self.depth_info = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.depth_frame is None:
            self.get_logger().info("Depth frame not yet received.")
            return

        if self.camera_info is None:
            self.get_logger().info("CameraInfo not yet received.")
            return

        results = self.model.track(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]

                point_3d = self.get_3d_coordinates(box)

                if point_3d is None:
                    continue

                point_camera = self.convert_to_point_stamped(point_3d, msg.header.stamp)

                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.target_frame,
                        point_camera.header.frame_id,
                        point_camera.header.stamp
                    )

                    point_base = do_transform_point(point_camera, transform)

                except tf2_ros.TransformException as ex:
                    self.get_logger().warn(f"Could not transform point to {self.target_frame}: {ex}")
                    continue

                self.get_logger().info(f"Detected {class_name} at 3D coordinates (camera frame): {point_3d}, (world frame): ({point_base.point.x}, {point_base.point.y}, {point_base.point.z})")

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.imshow("RealSense", frame)
        # cv2.waitKey(1)

    def depth_callback(self, msg):
        depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        self.depth_frame = depth_frame

        # # Visualize the depth image by normalizing it to 0-255 and converting to uint8        
        # max_depth = 20.0  # metres
        # depth_clipped = np.clip(depth_frame, 0, max_depth)  # Keep only values between 0 and max_depth
        # depth_visual = (depth_clipped / max_depth * 255).astype(np.uint8) # Convert to 0-255
        # depth_visual[depth_frame > max_depth] = 255 # Anything beyond max_depth becomes white

        # cv2.imshow("Depth", depth_visual)
        # cv2.waitKey(1)

    def image_info_callback(self, msg):
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)

    def depth_info_callback(self, msg):
        self.depth_info = msg

    def get_depth_of_box(self, box):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        window_size = 5

        half = window_size // 2

        region = self.depth_frame[
            v-half:v+half+1,
            u-half:u+half+1
        ]

        # Remove invalid values
        valid_depths = region[
            np.isfinite(region) &
            (region > 0)
        ]

        if len(valid_depths) == 0:
            return None

        return float(np.median(valid_depths))

    def get_3d_coordinates(self, box):
        if self.camera_info is None:
            self.get_logger().error("Camera model not initialized.")
            return None

        # Find the depth value of the bounding box
        depth = self.get_depth_of_box(box)

        if depth is None:
            self.get_logger().warn("No valid depth found for object.")
            return None

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        ray = self.camera_model.projectPixelTo3dRay((u, v))
        ray = np.array(ray)

        # Normalize the ray
        # ray /= np.linalg.norm(ray)

        # Scale the ray by the depth value
        point_3d = ray * depth

        return point_3d

    def convert_to_point_stamped(self, point_3d, stamp):
        point_stamped = PointStamped()
        point_stamped.header.stamp = stamp
        point_stamped.header.frame_id = self.camera_info.header.frame_id
        point_stamped.point.x = point_3d[0]
        point_stamped.point.y = point_3d[1]
        point_stamped.point.z = point_3d[2]
        return point_stamped

def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()