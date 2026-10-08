import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lindkvist',
    maintainer_email='lindkvist@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = project.object_detection:main',
            'tf_relay = project.tf_relay:main',
            'path_planner_node = project.navigation_node:main',
            'frontier_detector_node = project.frontier_detector_node:main',
            'path_follower = project.path_follower_node:main',
        ],
    },
)
