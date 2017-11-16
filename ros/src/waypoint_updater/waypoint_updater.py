#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
	self.cur_x = 0
	self.cur_y = 0
        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
	# waypoints[0].twist.twist.linear.x = 5
	rospy.logerr('pose')
	self.cur_x = msg.pose.position.x
	self.cur_y = msg.pose.position.y
	#rospy.logerr(msg.pose.position.x)
	#rospy.logerr(msg.pose.position.y)
	#rospy.logerr(msg.pose.orientation.x)
	#rospy.logerr(msg.pose.orientation.y)
	#rospy.logerr(msg.pose.orientation.z)
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
	pos_x_list = np.asarray([waypoint.pose.pose.position.x for waypoint in waypoints.waypoints])
	pos_y_list = np.asarray([waypoint.pose.pose.position.y for waypoint in waypoints.waypoints])

	pts = zip(pos_x_list-self.cur_x, pos_y_list-self.cur_y)
	pts_arr = np.asarray(pts)
	d2 = np.sum(pts_arr**2, axis=1)
	rospy.logerr(d2)
	ind_closest = np.argmin(d2)
	rospy.logerr("CLOSEST")
	rospy.logerr(ind_closest)
	rospy.logerr("Length")
	rospy.logerr(len(pos_x_list))
	final_waypoints = waypoints.waypoints[ind_closest:(ind_closest+100)]

	waypoints.waypoints = final_waypoints
	rospy.logerr('printing waypoints')
	rospy.logerr(len(waypoints.waypoints))
	for waypoint in waypoints.waypoints:
	  rospy.logerr('waypoint')
	  rospy.logerr(waypoint.pose.pose.position.x)
          rospy.logerr(waypoint.pose.pose.position.y)
	self.final_waypoints_pub.publish(waypoints)
        pass

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
