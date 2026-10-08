from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    project_share = get_package_share_directory('project')
    slam_toolbox_share = get_package_share_directory('slam_toolbox')

    slam_params_file = os.path.join(
        project_share,
        'config',
        'mapper_params_online_async.yaml'
    )

    slam_launch_file = os.path.join(
        slam_toolbox_share,
        'launch',
        'online_async_launch.py'
    )

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_launch_file),
        launch_arguments={
            'slam_params_file': slam_params_file
        }.items()
    )

    return LaunchDescription([

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
                '--child-frame-id', 'camera_0_color_optical_frame',
            ],
        ),

        Node(
            package='project',
            executable='tf_relay',
            name='tf_relay',
            parameters=[
                {'use_sim_time': True},
            ],
            output='screen',
        ),

        slam_toolbox_launch,

        Node(
            package='project',
            executable='camera_node',
            name='camera_node',
            parameters=[
                {'use_sim_time': True},
            ],
            output='screen',
        ),
    ])