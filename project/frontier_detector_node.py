#!/usr/bin/env python3
# frontier_detector_node.py

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
import numpy as np
import tf2_ros


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('visited_radius', 4.0)
        self.base_frame = self.get_parameter(
            'base_frame').get_parameter_value().string_value
        self.visited_radius = self.get_parameter(
            'visited_radius').get_parameter_value().double_value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.visited_grid = None
        self.visited_resolution = None
        self.visited_origin_x = None
        self.visited_origin_y = None

        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 1)

        self.map_pub = self.create_publisher(
            OccupancyGrid, 'frontiers', 1)

    def _resize_visited_grid(self, msg: OccupancyGrid) -> None:
        height = msg.info.height
        width = msg.info.width
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        same_map = (
            self.visited_grid is not None
            and self.visited_grid.shape == (height, width)
            and self.visited_resolution == resolution
            and self.visited_origin_x == origin_x
            and self.visited_origin_y == origin_y
        )
        if same_map:
            return

        resized = np.zeros((height, width), dtype=bool)
        if self.visited_grid is not None:
            old_y, old_x = np.nonzero(self.visited_grid)
            world_x = self.visited_origin_x + (old_x + 0.5) * self.visited_resolution
            world_y = self.visited_origin_y + (old_y + 0.5) * self.visited_resolution
            new_x = np.floor((world_x - origin_x) / resolution).astype(int)
            new_y = np.floor((world_y - origin_y) / resolution).astype(int)
            in_bounds = (
                (new_x >= 0) & (new_x < width)
                & (new_y >= 0) & (new_y < height)
            )
            resized[new_y[in_bounds], new_x[in_bounds]] = True

        self.visited_grid = resized
        self.visited_resolution = resolution
        self.visited_origin_x = origin_x
        self.visited_origin_y = origin_y

    def _mark_robot_nearby_visited(self, msg: OccupancyGrid) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id or 'map', self.base_frame, Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().debug(f'Robot pose unavailable for visited mask: {exc}')
            return

        resolution = msg.info.resolution
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        grid_x = (robot_x - self.visited_origin_x) / resolution
        grid_y = (robot_y - self.visited_origin_y) / resolution
        radius_cells = self.visited_radius / resolution

        x_min = max(int(np.floor(grid_x - radius_cells)), 0)
        x_max = min(int(np.ceil(grid_x + radius_cells)) + 1, msg.info.width)
        y_min = max(int(np.floor(grid_y - radius_cells)), 0)
        y_max = min(int(np.ceil(grid_y + radius_cells)) + 1, msg.info.height)
        if x_min >= x_max or y_min >= y_max:
            return

        cell_x = (np.arange(x_min, x_max) + 0.5 - grid_x) * resolution
        cell_y = (np.arange(y_min, y_max) + 0.5 - grid_y) * resolution
        nearby = (
            cell_y[:, np.newaxis] ** 2 + cell_x[np.newaxis, :] ** 2
            <= self.visited_radius ** 2
        )
        self.visited_grid[y_min:y_max, x_min:x_max] |= nearby

    def map_callback(self, msg: OccupancyGrid):
        if msg.info.width == 0 or msg.info.height == 0:
            return

        self._resize_visited_grid(msg)
        self._mark_robot_nearby_visited(msg)

        # Convert map data stored as a list to a 2D grid
        grid = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))

        free = (grid >= 0) & (grid <= 50)
        unk = (grid < 0)

        h, w = grid.shape
        neigh_unknown = np.zeros_like(unk, dtype=bool)
        not_visited = ~self.visited_grid

        def acc_shift(dst, src, dy, dx):
            if dy == -1 and dx == -1:
                dst[:-1, :-1] |= src[1:, 1:]
            elif dy == -1 and dx == 0:
                dst[:-1, :] |= src[1:, :]
            elif dy == -1 and dx == 1:
                dst[:-1, 1:] |= src[1:, :-1]
            elif dy == 0 and dx == -1:
                dst[:, :-1] |= src[:, 1:]
            elif dy == 0 and dx == 1:
                dst[:, 1:] |= src[:, :-1]
            elif dy == 1 and dx == -1:
                dst[1:, :-1] |= src[:-1, 1:]
            elif dy == 1 and dx == 0:
                dst[1:, :] |= src[:-1, :]
            elif dy == 1 and dx == 1:
                dst[1:, 1:] |= src[:-1, :-1]

        # 4-neighborhood, creat mask using np
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            acc_shift(neigh_unknown, unk, dy, dx)

        frontier = free & neigh_unknown & not_visited
        frontier_map = np.zeros_like(unk, dtype=np.int8)
        frontier_map[frontier] = 100

        # Publish frontier map
        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.info = msg.info
        out_msg.data = frontier_map.astype(np.int8).flatten().tolist()
        self.map_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
