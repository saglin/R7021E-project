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

            # RGB-D camera
            'rgb_topic': '/a200_0000/sensors/camera_0/color/image',
            'depth_topic': '/a200_0000/sensors/camera_0/depth/image',
            'camera_info_topic': '/a200_0000/sensors/camera_0/color/camera_info',

            'map_topic': '/map',

            # 'localization': 'false',
            # 'publish_octomap_full': 'true',

            'subscribe_rgbd': 'false',
            
            # 3D LiDAR point cloud
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/a200_0000/sensors/lidar3d_0/points',

            # 'subscribe_scan': 'true',
            # 'scan_topic': '/a200_0000/sensors/lidar3d_0/scan',

            # Use the robot's existing odometry
            'visual_odometry': 'true',
            'icp_odometry': 'true',

            # Synchronize RGB and depth messages approximately
            'approx_sync': 'true',

            # Show RTAB-Map's visualization
            # 'rtabmap_viz': 'true',
            'rtabmap_viz': 'false',

            'Icp/Strategy': '0',
            'Icp/Iterations': '10',
            'Icp/VoxelSize': '0.10',
            'Icp/MaxCorrespondenceDistance': '1.0',
            'Icp/MaxTranslation': '3.0',
            'Icp/PointToPlane': 'true',
            'Icp/PointToPlaneK': '20',
            'Icp/CorrespondenceRatio': '0.01',
            'Odom/Strategy': '0',

            'RGBD/NeighborLinkRefining': 'true',
            'RGBD/ProximityBySpace': 'true',
            'RGBD/ProximityByTime': 'false',
            'RGBD/ProximityPathMaxNeighbors': '10',
            'RGBD/OptimizeFromGraphEnd': 'false',

            'args': (
                '--delete_db_on_start '
                '--Grid/3D true '
                '--Grid/RayTracing true '
                '--Grid/NormalsSegmentation false '
                '--Grid/MaxGroundHeight 0.15 '
                '--Grid/MaxObstacleHeight 1.8 '
                '--Grid/CellSize 0.2 '
                '--Grid/Sensor 2 '
                '--Grid/Scan2dUnknownSpaceFilled true '
                '--Grid/RangeMax 8.0 '
                '--Reg/Force3DoF true '
                '--RGBD/CreateOccupancyGrid true '
            ),
        }.items(),
    )

    # Remap RTAB-Map's occupancy grid to /map
    rtabmap_group = GroupAction(
        actions=[
            SetRemap(
                src='/rtabmap/octomap_grid',
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