#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 #200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
	self.waypoints = None
	self.base_waypoints = None
	self.cur_x = 0
	self.cur_y = 0

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_wp_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        rospy.spin()

    def loop(self):
        rate = rospy.Rate(10) # 50Hz
        while not rospy.is_shutdown():
	  #rospy.logerr("running")
	  self.update_waypoints()
          rate.sleep()

    def pose_cb(self, msg):
        # TODO: Implement
	self.cur_x = msg.pose.position.x
	self.cur_y = msg.pose.position.y
	#rospy.logerr(msg.pose.position.x)
	#rospy.logerr(msg.pose.position.y)
	#rospy.logerr(msg.pose.orientation.x)
	#rospy.logerr(msg.pose.orientation.y)
	#rospy.logerr(msg.pose.orientation.z)
        pass

    def traffic_wp_cb(self, msg):
	ind =  msg.data
	#rospy.logerr("INDEX")
	#rospy.logerr(ind)
	i = 0
	if self.waypoints:
	  for waypoint in self.waypoints.waypoints:
	    if (ind != -1 and ind < 36 and ind > 18):
	      waypoint.twist.twist.linear.x = 0.0
	    elif (ind != -1):
	      waypoint.twist.twist.linear.x = 3.0
	    else:
	      waypoint.twist.twist.linear.x = 15.0
	    i = i + 1
	pass

    def waypoints_cb(self, msg):
        # TODO: Implement
	self.base_waypoints = msg.waypoints
	self.waypoints = msg
	for waypoint in self.base_waypoints:
	  waypoint.twist.twist.linear.x = 15.0

	while (self.cur_x == 0):
	   test = 1
	   #rospy.logerr('waiting')
	self.loop()
        pass

    def update_waypoints(self):
	pos_x_list = np.asarray([waypoint.pose.pose.position.x for waypoint in self.base_waypoints])
	pos_y_list = np.asarray([waypoint.pose.pose.position.y for waypoint in self.base_waypoints])

	pts = zip(pos_x_list-self.cur_x, pos_y_list-self.cur_y)
	pts_arr = np.asarray(pts)
	d2 = np.sum(pts_arr**2, axis=1)
	ind_closest = np.argmin(d2)
	num_waypoints = len(self.base_waypoints)
	last_waypoint = ind_closest+LOOKAHEAD_WPS
	final_waypoints = self.base_waypoints[ind_closest:min(last_waypoint, num_waypoints-1)]
	if (last_waypoint > num_waypoints):
	  final_waypoints = final_waypoints + self.base_waypoints[0:(last_waypoint%num_waypoints-1)]	  
	self.waypoints.waypoints = final_waypoints
	self.final_waypoints_pub.publish(self.waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
