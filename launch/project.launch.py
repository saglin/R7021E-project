from launch import LaunchDescription

from launch.actions import IncludeLaunchDescription, GroupAction

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource
)

from launch_ros.actions import Node, SetRemap

from ament_index_python.packages import (
    get_package_share_directory
)

import os


def generate_launch_description():

    project_share = get_package_share_directory('project')

    # --------------------------------------------------------------
    # RTAB-Map
    # --------------------------------------------------------------

    rtabmap_share = get_package_share_directory('rtabmap_launch')

    rtabmap_launch_file = os.path.join(
        rtabmap_share,
        'launch',
        'rtabmap.launch.py',
    )

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            rtabmap_launch_file
        ),
        launch_arguments={
            'use_sim_time': 'true',

            # Robot frames and odometry
            'frame_id': 'base_link',
            'odom_topic': '/a200_0000/platform/odom',

            # Keep publishing/updating the occupancy grid
            # even when the robot has not moved
            'map_always_update': 'true',
            'map_empty_ray_tracing': 'false',

            # RGB-D camera
            'rgb_topic': '/a200_0000/sensors/camera_0/color/image',
            'depth_topic': '/a200_0000/sensors/camera_0/depth/image',
            'camera_info_topic': '/a200_0000/sensors/camera_0/color/camera_info',

            'map_topic': '/map',

            'subscribe_rgbd': 'false',

            # 3D LiDAR point cloud
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/a200_0000/sensors/lidar3d_0/points',

            'subscribe_scan': 'true',
            'scan_topic': '/a200_0000/sensors/lidar3d_0/scan',

            # Use the robot's existing odometry
            'visual_odometry': 'false',
            'icp_odometry': 'false',

            # Synchronize RGB and depth messages approximately
            'approx_sync': 'true',

            # Show RTAB-Map's visualization
            'rtabmap_viz': 'true',
            # 'rtabmap_viz': 'false',

            'args': (
                '--delete_db_on_start '
                '--Grid/3D false '
                '--Grid/NormalsSegmentation false '
                '--Grid/MaxGroundHeight 0.15 '
                '--Grid/MaxObstacleHeight 1.8'
                '--Grid/Sensor 0 '
                '--Grid/Scan2dUnknownSpaceFilled true '
                '--Grid/RangeMax 8.0'
            ),
        }.items(),
    )

    # Remap RTAB-Map's occupancy grid to /map
    rtabmap_group = GroupAction(
        actions=[
            SetRemap(
                src='/rtabmap/grid_prob_map',
                dst='/map',
            ),
            rtabmap_launch,
        ]
    )

    # --------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------

    return LaunchDescription([

        # ----------------------------------------------------------
        # Relay simulator TF:
        # /a200_0000/tf        -> /tf
        # /a200_0000/tf_static -> /tf_static
        # ----------------------------------------------------------

        Node(
            package='project',
            executable='tf_relay',
            name='tf_relay',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),

        # ----------------------------------------------------------
        # Camera optical frame TF
        # ----------------------------------------------------------

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_optical_frame_tf',
            arguments=[
                '--x', '0',
                '--y', '0',
                '--z', '0',
                '--roll', '-1.57079632679',
                '--pitch', '0',
                '--yaw', '-1.57079632679',
                '--frame-id', 'camera_0_link',
                '--child-frame-id',
                'camera_0_color_optical_frame',
            ],
        ),

        # ----------------------------------------------------------
        # SLAM: RTAB-Map replaces SLAM Toolbox
        # ----------------------------------------------------------

        rtabmap_group,

        # ----------------------------------------------------------
        # Camera / YOLO
        # ----------------------------------------------------------

        Node(
            package='project',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),

        # ----------------------------------------------------------
        # Frontier detection
        # ----------------------------------------------------------

        Node(
            package='project',
            executable='frontier_detector_node',
            name='frontier_detector',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),

        # ----------------------------------------------------------
        # RRT exploration planner
        # ----------------------------------------------------------

        Node(
            package='project',
            executable='path_planner_node',
            name='path_planner_node',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),

        # ----------------------------------------------------------
        # Path follower
        # ----------------------------------------------------------

        Node(
            package='project',
            executable='path_follower',
            name='path_follower',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),
    ])