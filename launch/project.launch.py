from launch import LaunchDescription

from launch.actions import IncludeLaunchDescription

from launch.launch_description_sources import (
    PythonLaunchDescriptionSource
)

from launch_ros.actions import Node

from ament_index_python.packages import (
    get_package_share_directory
)

import os


def generate_launch_description():

    project_share = (
        get_package_share_directory('project')
    )

    slam_toolbox_share = (
        get_package_share_directory('slam_toolbox')
    )

    # --------------------------------------------------------------
    # SLAM Toolbox
    # --------------------------------------------------------------

    slam_params_file = os.path.join(
        project_share,
        'config',
        'mapper_params_online_async.yaml',
    )

    slam_launch_file = os.path.join(
        slam_toolbox_share,
        'launch',
        'online_async_launch.py',
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            slam_launch_file
        ),
        launch_arguments={
            'slam_params_file':
                slam_params_file,
        }.items(),
    )

    # --------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------

    return LaunchDescription([

        # ----------------------------------------------------------
        # Relay simulator TF
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
                '--frame-id',
                'camera_0_link',
                '--child-frame-id',
                'camera_0_color_optical_frame',
            ],
        ),

        # ----------------------------------------------------------
        # SLAM
        # ----------------------------------------------------------

        slam_launch,

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
                    'visited_radius': 2.0,
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

        # ----------------------------------------------------------
        # Scan fixer
        # ----------------------------------------------------------
        Node(
            package='project',
            executable='scan_fixer',
            name='scan_fixer',
            output='screen',
            parameters=[
                {
                    'use_sim_time': True,
                }
            ],
        ),
    ])