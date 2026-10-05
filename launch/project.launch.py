from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
            remappings=[
                ('/tf_static', '/a200_0000/tf_static'),
            ],
        ),

        Node(
            package='project',
            executable='camera_node',
            name='camera_node',
            remappings=[
                ('/tf', '/a200_0000/tf'),
                ('/tf_static', '/a200_0000/tf_static'),
            ],
        ),
    ])