"""Display the stock SO-101 reference; publishes virtual joint states only."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_share = Path(get_package_share_directory('pinchy_description'))
    bringup_share = Path(get_package_share_directory('pinchy_bringup'))
    robot_description = (
        description_share / 'urdf' / 'so101_reference.urdf'
    ).read_text()
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show virtual joint sliders.'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz visualization.'),
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            parameters=[{'robot_description': ParameterValue(
                robot_description, value_type=str)}], output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui', condition=IfCondition(gui),
            output='screen',
        ),
        Node(
            package='joint_state_publisher', executable='joint_state_publisher',
            condition=UnlessCondition(gui), output='screen',
        ),
        Node(
            package='rviz2', executable='rviz2', condition=IfCondition(rviz),
            arguments=['-d', str(bringup_share / 'config' / 'arm.rviz')],
            output='screen',
        ),
    ])
