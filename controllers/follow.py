#!/usr/bin/env python
'''

'''
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from pid import pid_controller
import matplotlib.pyplot as plt
import argparse
from lab4_color.msg import Point32Array

class follow:

    def __init__(self, speed, kp, ki, kd, use_rate=False):
        self.use_rate = use_rate
        self.meas_yaw_rate_list = []
        self.speed = speed
        self.pid = pid_controller(kp, ki, kd)

        rospy.init_node('follow')        
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, latch=True, queue_size=1)

        rospy.Subscriber('/dot', Point32Array, self.callback_dot, queue_size=1)  # Important to have queue size of 1 to avoid delays

    def callback_dot(self, msg):
        centroid = msg.points[0]
        x = -1 * (159.5 - centroid.x)
        print(x)

        #orientation = msg.pose.pose.orientation        
        #angular = msg.twist.twist.angular
        #_, _, current_yaw  = euler_from_quaternion([orientation.x, orientation.y,\
        #                                            orientation.z, orientation.w])
        #self.meas_yaw_list.append(current_yaw)     # Measure yaw angle from IMU + odometry
        #self.meas_yaw_rate_list.append(angular.z)  # Measure yaw rate from IMU


        #self.target_yaw_list.append(target_yaw)
        if self.use_rate:
            cmd_yaw_rate = self.pid.update_control_with_rate(0, x, self.meas_yaw_rate_list[-1])
        else:
            cmd_yaw_rate = self.pid.update_control(0, x)
        #self.cmd_yaw_rate_list.append(cmd_yaw_rate)

        msg_twist = Twist()
        msg_twist.linear.x = self.speed
        msg_twist.angular.z = cmd_yaw_rate
        self.pub_vel.publish(msg_twist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Yaw')
    parser.add_argument('--speed', type=float, default=2.5, help='speed')
    
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    if unknown: print('Unknown args:',unknown)

    f = follow(args.speed, 1.8, 0.0, 2.8, False)

    rospy.spin()