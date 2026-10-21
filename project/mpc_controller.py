'''
    A distance based path follower that uses MPC to incorporate reactive collision avoidance.
    
    Subscribers:
        '/a200_0000/platform/odom', Type: Odometry, robot odometry
        'path', Type: Path, the planned path to follow
        '/a200_0000/sensors/lidar2d_0/scan', Type: LaserScan, LiDAR data from the robot   
    Publishers:
        '/a200_0000/cmd_vel', Type: TwistStamped, desired velocities for the robot
'''


import do_mpc # Importing the MPC library
import casadi as ca
from casadi import vertcat, horzcat, DM # Importing specific functions from casadi for symbolic mathematics
import numpy as np # Importing numpy library for useful math operations
import matplotlib.pyplot as plt # For creating plots
import matplotlib.animation as animation # For creating animations
import math

ca.GlobalOptions.setNumpyMode(-1)

import rclpy                        # Import ROS2 Python client library
from rclpy.node import Node         # Import Node object from the rclpy library
from std_msgs.msg import String     # Import the string message-type from the std_msgs

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan


class MPC_Controller(Node):
    def __init__(self): # Constructor
        super().__init__('mpc_controller') # Initialize the node with a unique name.

        self.subscriber = self.create_subscription(Odometry, '/a200_0000/platform/odom', self.msg_callback, 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/a200_0000/cmd_vel', 1)
        self.path_sub = self.create_subscription(Path, 'path', self.path_callback, 1)
        self.scan_sub = self.create_subscription(LaserScan, '/a200_0000/sensors/lidar2d_0/scan', self.lidar_msg_callback, 1)

        self.initialize = True

        self.dt = 0.1 #in seconds    
        self.progressed_time = 0.0
        
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.desired_x = 0
        self.desired_y = 0
        self.desired_yaw = 0

        self.error_margin = 0.1

        self.robot_radius = 0.61

        self.prediction_horizon = 20
        self.step_time = 0.1  

        # Constraints
        self.minimum_velocity_x = -0.1 
        self.maximum_velocity_x = 0.8
        self.minimum_velocity_yaw = -0.8
        self.maximum_velocity_yaw = 0.8

        self.required_angle_diff = np.pi/180*60    #60 deg in rad
        self.obs_radius = 0.15
        self.safety_dist = self.robot_radius + self.obs_radius + 0.05     #0.05 is a margin so obstacles do not appear out of nowhere
        self.active_obs = []
        self.new_obs = []
        self.free_obs = ['obs1', 'obs2','obs3']
        self.path = []
        self.look_ahead_dist = 1.0

        self.mpc = self.create_model()
        self.mpc.setup()
        self.mpc.x0 = np.array([self.current_x, self.current_y, self.current_yaw])
        self.mpc.set_initial_guess()




    def msg_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        self.current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

        # Prints the current coordinates in the terminal
        self.get_logger().info(f"x = {self.current_x:.2f}, y = {self.current_y:.2f}, yaw = {self.current_yaw:.2f}")

    def get_xy(self, xr, yr, theta_r, dist, angle):
        dx = dist*np.cos(theta_r+angle)
        dy = dist*np.sin(theta_r+angle)
        return [xr+dx, yr+dy]

    def dist(self, xr, yr, x, y):
        dist = np.sqrt((x-xr)**2 + (y-yr)**2)
        return dist

    def get_angle(self, xr, yr, theta_r, x, y):
        dx = x - xr
        dy = y - yr
        return math.atan2(dy, dx)

    def lidar_msg_callback(self, msg): 
        # msg contains: angle_min, angle_max, angle_increment, time_increment, scan_time, range_min, range_max, ranges, intensities

        for index, distance in enumerate(msg.ranges):
            # If distance is within the LIDAR's hardware limits, add an obstacle if the robot is too close to an object
            if msg.range_min <= distance <= msg.range_max:
                raw_angle = msg.angle_min + index*msg.angle_increment
                point_angle = 2*np.pi-raw_angle

                if (distance < self.safety_dist) and (len(self.free_obs) > 0): 
                    add = True
                    if len(self.active_obs) > 0:
                        for obs in self.active_obs:
                            obs_angle = self.get_angle(self.current_x, self.current_y, self.current_yaw, obs[0], obs[1])
                            if abs(obs_angle-point_angle) < self.required_angle_diff:
                                add = False
                                break
                        
                    if add == True:
                        new = self.get_xy(self.current_x, self.current_y, self.current_yaw, distance, point_angle)
                        new_name = self.free_obs.pop(0)
                        new.append(new_name)
                        self.new_obs.append(new)



    def path_callback(self, msg): 
        path = []
        for pose in msg.poses:
            path.append((pose.pose.position.x, pose.pose.position.y))

        if len(path) > 0:

            for index in range(len(path)):
                point = path[index]
                distance = self.dist(self.current_x, self.current_y, point[0], point[1])
                if distance > self.look_ahead_dist:
                    self.desired_x = point[0]
                    self.desired_y = point[1]
                    self.desired_yaw = self.get_angle(self.current_x, self.current_y, self.current_yaw, point[0], point[1])

                    self.path = path[index:]
                    break
        else:
            self.desired_x = self.current_x
            self.desired_y = self.current_y
            self.desired_yaw = self.current_yaw


    def timer_callback(self): 
        msg = TwistStamped()

        self.progressed_time += self.dt

        error_x = self.desired_x - self.current_x
        error_y = self.desired_y - self.current_y

        distance_error = math.sqrt(error_x**2 + error_y**2)     # Euclidian distance (actual distance)

        if (distance_error < self.error_margin) and (len(self.path) > 0):
            new_pos = self.path.pop(0)
            self.desired_x = new_pos[0]
            self.desired_y = new_pos[1]
            self.desired_yaw = self.get_angle(self.current_x, self.current_y, self.current_yaw, new_pos[0], new_pos[1])


        tvp = self.mpc.get_tvp_template()
 
        
        tvp_fun = lambda t_now: self.tvp_fun(t_now, tvp, callback=True)
        self.mpc.set_tvp_fun(tvp_fun) 

        p = np.array([self.current_x, self.current_y, self.current_yaw])

        u = self.mpc.make_step(p).flatten()

        msg.twist.linear.x = float(np.clip(u[0], self.minimum_velocity_x, self.maximum_velocity_x))

        msg.twist.angular.z = float(np.clip(u[1], self.minimum_velocity_yaw, self.maximum_velocity_yaw))

        msg.header.stamp = self.get_clock().now().to_msg()

        self.vel_pub.publish(msg)
        

    def quaternion_to_yaw(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )
        
    def tvp_fun(self, t_now, tvp_template, callback = False): 
        if self.initialize:
            #initialize all obstacles to r=0
            for k in range(self.prediction_horizon+1):
                tvp_template['_tvp',k,'obs1_r'] = 0.0
                tvp_template['_tvp',k,'obs1_x'] = 0.0
                tvp_template['_tvp',k,'obs1_y'] = 0.0

                tvp_template['_tvp',k,'obs2_r'] = 0.0
                tvp_template['_tvp',k,'obs2_x'] = 0.0
                tvp_template['_tvp',k,'obs2_y'] = 0.0

                tvp_template['_tvp',k,'obs3_r'] = 0.0
                tvp_template['_tvp',k,'obs3_x'] = 0.0
                tvp_template['_tvp',k,'obs3_y'] = 0.0
            
            self.initialize = False
            self.get_logger().info(f"obstacles initialized")
        else:
            if len(self.active_obs) > 0:
                for obs in self.active_obs:
                    distance = self.dist(self.current_x, self.current_y, obs[0], obs[1])
                    if distance > self.safety_dist:
                        #remove obstacle
                        for k in range(self.prediction_horizon+1):
                            tvp_template['_tvp',k,obs[2]+'_r'] = 0.0
                        self.active_obs.remove(obs)
                        self.free_obs.append(obs[2])
                        self.get_logger().info(f"obstacle: {obs[2]} removed")
                        
            if len(self.new_obs) > 0:
                for obs in self.new_obs:
                    #add obstacle
                    for k in range(self.prediction_horizon+1):
                        tvp_template['_tvp',k,obs[2]+'_r'] = self.obs_radius
                        tvp_template['_tvp',k,obs[2]+'_x'] = obs[0]
                        tvp_template['_tvp',k,obs[2]+'_y'] = obs[1]
                    self.new_obs.remove(obs)
                    self.active_obs.append(obs)
                    self.get_logger().info(f"obastacle {obs[2]} added at: x = {obs[0]:.2f}, y = {obs[1]:.2f}")
        if callback == True:
            for k in range(len(tvp_template['_tvp'])):
                tvp_template['_tvp',k,'xdes'] = self.desired_x
                tvp_template['_tvp',k,'ydes'] = self.desired_y
                tvp_template['_tvp',k,'yawdes'] = self.desired_yaw
        
        return tvp_template

    def create_model(self):
        model_type = 'continuous'
        sys_model = do_mpc.model.Model(model_type)

        x = sys_model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        y = sys_model.set_variable(var_type='_x', var_name='y', shape=(1,1))
        yaw = sys_model.set_variable(var_type='_x', var_name='yaw', shape=(1,1))

        # Underactuated system (we cannot control velocity in y direction directly)
        vx = sys_model.set_variable(var_type='_u', var_name='vx')
        vyaw = sys_model.set_variable(var_type='_u', var_name='vyaw')

        x_des = sys_model.set_variable(var_type='_tvp', var_name='xdes', shape=(1,1))
        y_des = sys_model.set_variable(var_type='_tvp', var_name='ydes', shape=(1,1))
        yaw_des = sys_model.set_variable(var_type='_tvp', var_name='yawdes', shape=(1,1))

        obs1_x = sys_model.set_variable(var_type='_tvp', var_name='obs1_x', shape=(1,1))
        obs1_y = sys_model.set_variable(var_type='_tvp', var_name='obs1_y', shape=(1,1))
        obs1_r = sys_model.set_variable(var_type='_tvp', var_name='obs1_r', shape=(1,1))

        obs2_x = sys_model.set_variable(var_type='_tvp', var_name='obs2_x', shape=(1,1))
        obs2_y = sys_model.set_variable(var_type='_tvp', var_name='obs2_y', shape=(1,1))
        obs2_r = sys_model.set_variable(var_type='_tvp', var_name='obs2_r', shape=(1,1))

        obs3_x = sys_model.set_variable(var_type='_tvp', var_name='obs3_x', shape=(1,1))
        obs3_y = sys_model.set_variable(var_type='_tvp', var_name='obs3_y', shape=(1,1))
        obs3_r = sys_model.set_variable(var_type='_tvp', var_name='obs3_r', shape=(1,1))

        sys_model.set_rhs('x', vx*ca.cos(yaw))
        sys_model.set_rhs('y', vx*ca.sin(yaw))
        sys_model.set_rhs('yaw', vyaw)

        sys_model.setup()

        mpc = do_mpc.controller.MPC(sys_model)

        setup_mpc = {
            'n_horizon': self.prediction_horizon,
            't_step': self.step_time,
            'n_robust': 0,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        Q = np.diag([1.0, 1.0, 1.0])   # state weights (x, y, yaw)
        R = np.diag([0.1, 0.1])   # input weights (vx, vyaw)

        x = sys_model.x['x']
        y = sys_model.x['y']
        yaw = sys_model.x['yaw']

        x_des = sys_model.tvp['xdes']
        y_des = sys_model.tvp['ydes']
        yaw_des = sys_model.tvp['yawdes']

        gx = x_des - x
        gy = y_des - y

        dist2 = gx**2 + gy**2

        yaw_error_cost = 1.0 - ca.cos(yaw - yaw_des)

        heading_w = dist2 / (dist2 + 0.25)

        lterm = (
            5.0 * dist2
            + 2.0 * heading_w * yaw_error_cost
        )

        mterm = (
            15.0 * dist2
            + 5.0 * heading_w * yaw_error_cost
        )

        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(vx=0.1, vyaw=0.1)
        
        # Input constraints
        mpc.bounds['lower','_u','vx'] = self.minimum_velocity_x
        mpc.bounds['upper','_u','vx'] = self.maximum_velocity_x
        mpc.bounds['lower','_u','vyaw'] = self.minimum_velocity_yaw
        mpc.bounds['upper','_u','vyaw'] = self.maximum_velocity_yaw

        # Obstacle constraints
        mpc.set_nl_cons(
            'obs1',
            obs1_r**2-((sys_model.x['x']-obs1_x)**2+(sys_model.x['y']-obs1_y)**2),
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=1e3
        )

        mpc.set_nl_cons(
            'obs2',
            obs2_r**2-((sys_model.x['x']-obs2_x)**2+(sys_model.x['y']-obs2_y)**2),
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=1e3
        )

        mpc.set_nl_cons(
            'obs3',
            obs3_r**2-((sys_model.x['x']-obs3_x)**2+(sys_model.x['y']-obs3_y)**2),
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=1e3
        )
        
        
            
        tvp = mpc.get_tvp_template()

        tvp_fun = lambda t_now: self.tvp_fun(t_now, tvp)
        mpc.set_tvp_fun(tvp_fun) # might need tvp_fun()

        return mpc


def main(args=None): # Main Function
    rclpy.init(args=args) # Initialize the ros client
    node = MPC_Controller() # Declare an object of the Receiver class.
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.

if __name__=='__main__': # Entry point
    main() # Call the main function from the entry point.
