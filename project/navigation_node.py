#!/usr/bin/env python3

import math
from typing import Optional, Tuple, List
import numpy as np
from skimage.draw import line
import tf2_ros
import random
import copy

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Path, OccupancyGrid
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time
from vision_msgs.msg import Detection3DArray
from scipy.spatial import KDTree
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point



def quat_to_yaw(q: Quaternion) -> float:
    """Extract planar yaw (rad) from a Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert planar yaw (rad) to Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class PathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('path_planner_node')

        # Parameters
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('frontier_topic', 'frontiers')
        self.declare_parameter('path_topic', 'path')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 0.5)
        self.detected_objects_topic = '/detected_objects'

        map_topic: str = self.get_parameter('map_topic').get_parameter_value().string_value
        frontier_topic: str = self.get_parameter('frontier_topic').get_parameter_value().string_value
        path_topic: str = self.get_parameter('path_topic').get_parameter_value().string_value
        self.global_frame: str = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame: str = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tf_timeout = Duration(seconds=self.get_parameter('tf_timeout_sec').get_parameter_value().double_value)

        default_qos = QoSProfile(depth=10)

        # Publishers (add path publisher here)
        self.path_pub = self.create_publisher(Path, path_topic, 1)
        self.tree_pub = self.create_publisher(Marker, 'tree_nodes', 1)

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self._on_map, default_qos)
        self.frontier_sub = self.create_subscription(OccupancyGrid, frontier_topic, self._on_frontier, default_qos)
        self.detected_objects = self.create_subscription(Detection3DArray, self.detected_objects_topic, self._coordinates_detected_object, default_qos)


        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self._latest_map: Optional[OccupancyGrid] = None
        self._latest_frontier: Optional[OccupancyGrid] = None

        self.get_logger().info('PathPlannerNode initialized.')

        self.tree_iterations = 3200
        self.occupancy_threshold = 80
        self.lidar_range = 4 # 3.5
        self.inflation_radius = 0.7
        self.distance_gain = 1.0
        self.ig_coeff = 5.0
        self.minimum_travel_distance = 0.1
#👉🏻
        self.detected_object_gain = None
        self.detected_object_list = []
# project addition ⚠️: adding dictionary for idetifying certain objects. Where Key is going to be the 
# detcted object and the values will be a list of ranked objects that the key can be close to.
        self.object_value_list = {"person":100,"car":50, "truck":50, "bench":60, "backpack":40, "suitcase":40, "chair":70, "couch":70, "table":75, "toilet":25}


        self.minimum_step_length = 5    # in cm
        self.maximum_step_length = 50   # in cm

        self.current_tree = None
        self.current_leaf_node = None
        self.current_path = None 
        self.max_path_deviation = 0.5  # meters

        # self.frontier_percentage_threshold = 0.01  # 1%
        # self.exploration_complete = False


    # ----------------- TF Helper -----------------

    def get_robot_pose(self, target_frame: Optional[str] = None, source_frame: Optional[str] = None
                       ) -> Optional[Tuple[float, float, float]]:
        """
        Lookup TF to get robot pose (x, y, yaw) in target_frame.
        Returns None if not available within timeout.
        """
        tgt = target_frame or self.global_frame
        src = source_frame or self.base_frame

        try:
            transform = self.tf_buffer.lookup_transform(
                tgt, src, Time(), timeout=self.tf_timeout
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup {tgt} <- {src} failed: {e}')
            return None

        t = transform.transform.translation
        r = transform.transform.rotation
        yaw = quat_to_yaw(r)

        return (t.x, t.y, yaw)

    # ----------------- Callbacks -----------------

    def _on_frontier(self, msg: OccupancyGrid) -> None:
        """Store the latest frontier map (could be used by your planner)."""
        self._latest_frontier = msg

        # # Count total cells in the frontier map
        # total_cells = msg.info.width * msg.info.height

        # # makes sure that the frontier map isn't empty
        # if total_cells == 0:
        #     return

        # # Count the frontier cells
        # frontier_count = 0
        # for cell in msg.data:
        #     if cell > 0:
        #         frontier_count += 1

        # # Calculate the percentage of the map that is frontier
        # frontier_percentage = frontier_count / total_cells

        # # Stop planning when frontiers occupy 1% or less of the map
        # if frontier_percentage <= self.frontier_percentage_threshold:
        #     self.exploration_complete = True

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Trigger planning when a new map arrives."""
        # self.get_logger().info("Map Received")
        self._latest_map = msg

        pose = self.get_robot_pose()
        if pose is None:
            return
        
        x, y, yaw = pose

        if self.current_path is None:
            self.plan_path((x, y, yaw),msg,self._latest_frontier)

        else:
            if self.check_if_goal_is_reached(pose):
                self.current_path = None

            elif self.check_if_collision_on_path(msg, pose):
                self.current_path = None
                self.plan_path(pose, msg, self._latest_frontier)
        
    
        
        

    def _coordinates_detected_object(self, msg):
        self.get_logger().info(f"Detection frame: {msg.header.frame_id}")
        for detection in msg.detections:
            # Label
            if len(detection.results) == 0:
                continue
            #confidance score for label
            detected_confidance = detection.results[0].hypothesis.score
            if detected_confidance > 0.8:
                #label
                detected_label = detection.results[0].hypothesis.class_id
                
                object_point = PointStamped()
                object_point.header = msg.header
                object_point.point = detection.bbox.center.position

                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.global_frame,
                        object_point.header.frame_id,
                        Time(),
                        timeout=self.tf_timeout
                    )

                    transformed_point = do_transform_point(object_point, transform)

                except Exception as e:
                    self.get_logger().warn(f"Object transformation failed: {e}")
                    continue

                x = transformed_point.point.x
                y = transformed_point.point.y

                # id of the object, ALWAYS uniqe
                dected_id = detection.id

                
                existing_index = None
                for idx, (_, _, _, id_in_list) in enumerate(self.detected_object_list):
                    if dected_id == id_in_list:
                        existing_index = idx
                        break

                if existing_index is not None:
                    self.detected_object_list[existing_index] = (x, y, detected_label, dected_id)
                else:
                    self.detected_object_list.append((x, y, detected_label, dected_id))

           
    

    # ----------------- Planning -----------------

    def plan_path(self,
                  start: Tuple[float, float, float],
                  map_msg: OccupancyGrid,
                  frontier_msg: Optional[OccupancyGrid],
                  ):
        """
        Use this function to plan the path
        """
        # # Stop planning when exploration is complete
        # if self.exploration_complete:
        #     self.get_logger().info("Exploration complete")
        #     return      

        # self.get_logger().info("Planning path...")
    
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        min_x = map_msg.info.origin.position.x
        min_y = map_msg.info.origin.position.y

        max_x = min_x + width * resolution
        max_y = min_y + height * resolution

        grid_array = map_msg.data


        # Make the occupancy map 2D
        grid_2d = []
        for x_index in range(width):
            grid_2d.append([])
            for y_index in range(height):
                grid_2d[x_index].append(grid_array[y_index * width + x_index])

        inflated_grid = np.asarray(self.inflate_map(grid_2d, resolution))

        

        # 1. Initialize a new RRT tree
        tree = Tree(start[0], start[1])

        # Initialize KDTree for nearest-neighbor search
        kdtree = KDTree(tree.coords)
        kdtree_size = len(tree.coords)
        rebuild_interval = 25
        
        
        for i in range(self.tree_iterations):
            # 2. Random point in M
            random_x = random.random() * (max_x - min_x) + min_x
            random_y = random.random() * (max_y - min_y) + min_y

            # 3. Find the closest node/vertex to rho within N, denoted N_closest
            if len(tree.coords) - kdtree_size >= rebuild_interval:
                kdtree = KDTree(tree.coords)
                kdtree_size = len(tree.coords)

            #MODIFIED CODE
            _distance, closest_index = kdtree.query([random_x, random_y])
            new_coords = tree.coords[kdtree_size:]

            if len(new_coords) > 0:
                new_coords_array = np.asarray(new_coords)
                distances = np.sum((new_coords_array - [random_x, random_y])**2, axis=1)
                new_closest_index = np.argmin(distances)
                # Squared distance to that node
                new_closest_distance = distances[new_closest_index]
                # Compare closest new node against KDTree result
                if new_closest_distance < _distance**2:
                    closest_index = kdtree_size + new_closest_index

            N_closest = tree.nodes[closest_index]

            # OLD
            # min_distance = math.inf
            # N_closest = tree.root
            # for node in tree.nodes:
            #     distance = math.sqrt((random_x-node.x)**2 + (random_y-node.y)**2)

            #     if distance < min_distance:
            #         N_closest = node
            #         min_distance = distance

            # 4. Take a "step" from N_closest to rho
            angle_to_point = math.atan2(random_y - N_closest.y, random_x - N_closest.x) # Calculates the angle between the x-axis and the line from N_closest and rho
            step_length = random.randint(self.minimum_step_length, self.maximum_step_length) / 100   # Calculates a random step length
            rho_c_x = N_closest.x + math.cos(angle_to_point) * step_length 
            rho_c_y = N_closest.y + math.sin(angle_to_point) * step_length

            # 5. Check if rho_c is occupied
            # First, calculate the index of the grid square rho_c is in
            rho_x_index = math.floor((rho_c_x - min_x) / resolution)
            rho_y_index = math.floor((rho_c_y - min_y) / resolution)

            # Makes sure that rho_c is within the map
            if not (0 <= rho_x_index < width and 0 <= rho_y_index < height):
                # self.get_logger().info("Index error")
                continue

            # Checks the occupancy status of rho_c
            if inflated_grid[rho_x_index][rho_y_index] < self.occupancy_threshold:

                # Checks occapancy of every square between N_closest and rho_c
                if not self.collision_check(N_closest.x, N_closest.y, rho_c_x, rho_c_y, inflated_grid, min_x, min_y, resolution):
                    distance = math.sqrt((rho_c_x - N_closest.x)**2 + (rho_c_y - N_closest.y)**2)
                    tree.add_node(rho_c_x, rho_c_y, N_closest, distance)
                
        # Checks that a frontier map exists. It might not exist for the very first iteration of the 
        # path planning since the frontier construction runs at the same time as this path planning.
        if frontier_msg is None:
            return 

        # Gets the data about the frontier map. The size might not neccessarily be the same as
        # the occupancy map.
        frontier_grid_array = frontier_msg.data
        frontier_width = frontier_msg.info.width
        frontier_height = frontier_msg.info.height
        frontier_resolution = frontier_msg.info.resolution
        frontier_min_x = frontier_msg.info.origin.position.x
        frontier_min_y = frontier_msg.info.origin.position.y

        frontier_max_x = min_x + width * resolution
        frontier_max_y = min_y + height * resolution

        # Make the frontier grid into a 2D array
        frontier_grid_2d = []
        for x_index in range(frontier_width):
            frontier_grid_2d.append([])
            for y_index in range(frontier_height):
                frontier_grid_2d[x_index].append(frontier_grid_array[y_index * frontier_width + x_index])

    
        # Calculate the heuristic values of all the leaf nodes in the tree
        heuristic_values = []
        
        for leaf_node in tree.leaf_nodes:

            heuristic_value = self.calculate_heuristic(leaf_node, inflated_grid, min_x, min_y, resolution, frontier_grid_2d, frontier_min_x, frontier_min_y, frontier_resolution)
            heuristic_values.append(heuristic_value)



        # Find the best heuristic value and which node it corresponds to
        heuristic_value = min(heuristic_values)
        index = heuristic_values.index(heuristic_value)
        best_leaf_node = tree.leaf_nodes[index]

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        # Extract the path from the leaf node to the root (the robot's position)
        node = best_leaf_node
        while node is not None:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = "map"

            pose.pose.position.x = node.x
            pose.pose.position.y = node.y

            path_msg.poses.insert(0, pose)
            node = node.parent

        # Publish the path
        self.path_pub.publish(path_msg)
        self.current_path = path_msg

        # Save the new tree and the goal node
        self.current_tree = tree
        self.current_leaf_node = best_leaf_node

        self.publish_tree(self.current_tree)
        

        return path_msg

# 👇🏼 new since 15/10:
    def check_if_goal_is_reached(self, robot_pose):
        #1. goal point, end of the path
        goal_point = self.current_path.poses[-1].pose.position
        g_x, g_y, = goal_point.x, goal_point.y

        #2. the robots current pose
        c_x, c_y, c_yaw = robot_pose

        #3. calculate the euclidian distance between current position and the goal position (end of the path)
        distance = math.sqrt((g_x - c_x)**2 + (g_y - c_y)**2)

        #4. check if that is with the threshold
        if distance < self.minimum_travel_distance:
            #5. if the goal is reached
            return True
        else:
            return False


    def check_if_collision_on_path(self, map_msg, robot_pose):
        # 1. Get current path
        path_points = self.current_path.poses
        
        if len(path_points) < 2:
            return True

        robot_x, robot_y, _ = robot_pose
        closest_segment_index = 0
        minimum_distance = float("inf")

        for i in range(len(path_points) - 1):
            point_a = path_points[i].pose.position
            point_b = path_points[i + 1].pose.position
            
            dx = point_b.x - point_a.x
            dy = point_b.y - point_a.y
            segment_length_squared = dx**2 + dy**2

            if segment_length_squared == 0:
                continue
            t = ((robot_x - point_a.x) * dx +
                (robot_y - point_a.y) * dy) / segment_length_squared
            t = max(0.0, min(1.0, t))


            # Calculate the projected point on the segment
            closest_x = point_a.x + t * dx
            closest_y = point_a.y + t * dy

            # Calculate distance from robot to projected point
            distance = math.sqrt(
                (robot_x - closest_x)**2 +
                (robot_y - closest_y)**2
            )

            if distance < minimum_distance:
                minimum_distance = distance
                closest_segment_index = i

        # Check whether robot has deviated too far from the path
        if minimum_distance > self.max_path_deviation:
            return True

        # 2. Get occupancy grid
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution

        min_x = map_msg.info.origin.position.x
        min_y = map_msg.info.origin.position.y



        grid_2d = np.array(map_msg.data).reshape(height, width).T

        # 3. Inflate occupancy grid
        inflated_grid = np.asarray(
            self.inflate_map(grid_2d, resolution)
        )

        # 4. Iterate through consecutive points
        for i in range(closest_segment_index, len(path_points) - 1):
            point_a = path_points[i].pose.position
            point_b = path_points[i + 1].pose.position

            if i == closest_segment_index:
                start_x = robot_x
                start_y = robot_y
            else:
                start_x = point_a.x
                start_y = point_a.y

            # 5. Check collision between two points
            collision = self.collision_check(start_x, start_y, point_b.x, point_b.y, inflated_grid, min_x, min_y, resolution)

            # 6. Return True if blocked
            if collision:
                return True

        # 7. No collision found
        return False
# 👆🏻 new since 15/10:
        
        

    # Checks if any grid squares between point_a and point_b are occupied
    
    def collision_check(self, point_a_x, point_a_y, point_b_x, point_b_y, occupancy_grid, min_x, min_y, resolution):

        # Convert world coordinates to grid indices
        x0 = math.floor((point_a_x - min_x) / resolution)
        y0 = math.floor((point_a_y - min_y) / resolution)
        x1 = math.floor((point_b_x - min_x) / resolution)
        y1 = math.floor((point_b_y - min_y) / resolution)

        # Get the cells along the line
        xx, yy = line(x0, y0, x1, y1)

        # Check map boundaries
        width = len(occupancy_grid)
        height = len(occupancy_grid[0])

        if np.any((xx < 0) | (xx >= width) |
                (yy < 0) | (yy >= height)):
            return True

        # Check for occupied cells
        return bool(np.any(occupancy_grid[xx, yy] >= self.occupancy_threshold))


#👉🏻
    def calculate_heuristic(self, leaf_node, occupancy_grid, min_x, min_y, resolution, frontier_grid, frontier_min_x, frontier_min_y, frontier_resolution):
        
        information_gain = self.calculate_information_gain(leaf_node, occupancy_grid, min_x, min_y, resolution, frontier_grid, frontier_min_x, frontier_min_y, frontier_resolution)
        heuristic_value = self.distance_gain * leaf_node.distance_to_root - self.ig_coeff * information_gain
        return heuristic_value
#👉🏻
    def calculate_current_heuristic(self, leaf_node, occupancy_grid, min_x, min_y, resolution, frontier_grid, frontier_min_x, frontier_min_y, frontier_resolution, robot_x, robot_y):
        
        information_gain = self.calculate_information_gain(leaf_node, occupancy_grid, min_x, min_y, resolution, frontier_grid, frontier_min_x, frontier_min_y, frontier_resolution)

        node = leaf_node
        distance = 0
        while node is not None:
            robot_node_distance = math.sqrt((node.x - robot_x)**2 + (node.y - robot_y)**2)

            if robot_node_distance < node.distance_to_parent:
                distance += robot_node_distance
                break

            else:
                distance += node.distance_to_parent

                node = node.parent

        heuristic_value = self.distance_gain * leaf_node.distance_to_root - self.ig_coeff * information_gain
        return heuristic_value

#👉🏻
    def calculate_information_gain(self, leaf_node, occupancy_grid, min_x, min_y, resolution, frontier_grid, frontier_min_x, frontier_min_y, frontier_resolution):

        information_gain = 0

        for index_x in range(len(frontier_grid)):
            for index_y in range(len(frontier_grid[index_x])):
                if frontier_grid[index_x][index_y] > 0: # Checks if the square is a frontier

                    x = index_x * frontier_resolution + frontier_resolution / 2 + frontier_min_x
                    y = index_y * frontier_resolution + frontier_resolution / 2 + frontier_min_y

                    # Calculates the distance from the frontier to the leaf node 
                    distance = math.sqrt((x - leaf_node.x)**2 + (y - leaf_node.y)**2)

                    # If the distance is within the lidar range it might be discovered if the 
                    # robot moves to the leaf. 
                    if distance < self.lidar_range:
                        if not self.collision_check(leaf_node.x, leaf_node.y, x, y, occupancy_grid, min_x, min_y, resolution):
                            information_gain += 1

        #for x,y,detected_objected,id in self.detected_object_list: 
        #    object_value = self.object_value_list[detected_objected]
        #    distance = math.sqrt((x - leaf_node.x)**2 + (y - leaf_node.y)**2)
        #    if distance < self.lidar_range:
        #        information_gain += object_value

        # only add object value of the highest object value in the detected object list that is within the lidar range
        max_object_value = 0
        for x,y,detected_objected,id in self.detected_object_list:
            object_value = self.object_value_list.get(detected_objected, 0)
            distance = math.sqrt((x - leaf_node.x)**2 + (y - leaf_node.y)**2)
            if distance < self.lidar_range:
                max_object_value = max(max_object_value, object_value)

        information_gain += max_object_value
        

        return information_gain

    


    def publish_tree(self, tree):
        marker = Marker()

        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "rrt_tree"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Line width in meters
        marker.scale.x = 0.02

        # Color
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Prevent old tree from remaining indefinitely
        marker.lifetime.sec = 0

        for node in tree.nodes:
            if node.parent is None:
                continue

            p1 = Point()
            p1.x = node.parent.x
            p1.y = node.parent.y
            p1.z = 0.05

            p2 = Point()
            p2.x = node.x
            p2.y = node.y
            p2.z = 0.05

            marker.points.append(p1)
            marker.points.append(p2)

        self.tree_pub.publish(marker)

    
    # Inflates the occupied squares of the occupancy grid.
    def inflate_map(self, grid, resolution):
        inflated_grid = copy.deepcopy(grid)
        inflation_cells = math.ceil(self.inflation_radius / resolution)
        width = len(grid)
        height = len(grid[0])

        # Iterates through the old occupancy grid and changes the occupancy of the squares around 
        # all the occupied squares.
        for x in range(width):
            for y in range(height):
                if grid[x][y] >= self.occupancy_threshold:  # Checks if the square is occupied
                    # Iterates through the cells around the occupied square
                    for dx in range(-inflation_cells, inflation_cells + 1):
                        for dy in range(-inflation_cells, inflation_cells + 1):
                            nx = x + dx
                            ny = y + dy

                            # Checks that the surrounding square is within the map
                            if 0 <= nx < width and 0 <= ny < height:
                                # Only changes nearby squares if they are within a circular radius 
                                # of the original square
                                if dx**2 + dy**2 <= inflation_cells**2:
                                    inflated_grid[nx][ny] = 100

        return inflated_grid


class Tree():
    def __init__(self, robot_x, robot_y):
        self.root = Tree_Nodes(robot_x, robot_y)
        self.nodes = [self.root]
        self.leaf_nodes = [self.root]
        self.coords = [(robot_x, robot_y)] #new code

    def add_node(self, x, y, parent, distance_to_parent=0.0, children=None):
        new_node = Tree_Nodes(x, y, parent, distance_to_parent, children)
        self.nodes.append(new_node)
        self.coords.append((x, y)) #new code
        self.leaf_nodes.append(new_node)
        if parent in self.leaf_nodes:
            self.leaf_nodes.remove(parent)
        parent.child.append(new_node)


class Tree_Nodes():
    def __init__(self, x, y, parent=None, distance_to_parent=0.0, children=None):
        self.x = x
        self.y = y
        self.parent = parent  # For the root node the parent is None
        if parent is None:
            self.distance_to_root = 0.0
            self.distance_to_parent = 0.0
        else:
            self.distance_to_root = distance_to_parent + parent.distance_to_root
            self.distance_to_parent = distance_to_parent
        self.child = []

        if children != None:
            for child in children:
                self.append(child)

        
def main() -> None:
    rclpy.init()
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()