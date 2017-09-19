#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32 # used for traffic_waypoint subscribtion

import math

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

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        
        # Subriber for /traffic_waypoint
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        # TODO: Subscriber for /obstacle_waypoint
        #rospy.Subscriber('/obstacle_waypoint', ???, self.obstacle_cb, queue_size=1)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        # Array to store waypoints data; is later added to lane object
        self.waypoints_data = []

        rospy.spin()

    def pose_cb(self, msg):

    		# Lane object (node output)
        myLane = Lane()

        # Current vehicle position
        current_x = msg.pose.position.x
        current_y = msg.pose.position.y

        # Search nearest waypoint
        min_dist = 9999999
        nearest_index = -1
        for i in range(len(self.waypoints_data)):
        	pos_x = self.waypoints_data[i].pose.pose.position.x
        	pos_y = self.waypoints_data[i].pose.pose.position.y
        	if (self.dst(pos_x, pos_y, current_x, current_y) < min_dist):
        		min_dist = self.dst(pos_x, pos_y, current_x, current_y)
        		nearest_index = i
        		#print("min_dist ", min_dist) # just for debugging

        # Nearest waypoint could also lie behind the car... if so: consider the next waypoint in line
        increase_index = False
        nearest_x = self.waypoints_data[nearest_index].pose.pose.position.x
        nearest_y = self.waypoints_data[nearest_index].pose.pose.position.y
        nearest_next_x = self.waypoints_data[nearest_index+1].pose.pose.position.x
        nearest_next_y = self.waypoints_data[nearest_index+1].pose.pose.position.y
        if (((nearest_next_x-nearest_x>0.0) and (current_x>nearest_x)) or ((nearest_next_y-nearest_y>0.0) and (current_y>nearest_y)) or
        	((nearest_next_x-nearest_x<0.0) and (current_x<nearest_x)) or ((nearest_next_y-nearest_y<0.0) and (current_y<nearest_y))):
        	increase_index = True
        if (increase_index):
        	nearest_index+=1

        # Provide info to lane object; ensure not to exceed max index of waypoints_data
        myLane.waypoints = self.waypoints_data[nearest_index:min(nearest_index+LOOKAHEAD_WPS,len(self.waypoints_data)-nearest_index-1)]
        self.final_waypoints_pub.publish(myLane)

        # DEBUGGING
        #print("current_x | current_y =", current_x, " | ", current_y)
        #print("min_dist = ", min_dist)
        #print("waypoints_data[nearest_index].x ", self.waypoints_data[nearest_index].pose.pose.position.x)
        #print("myLane.waypoints = ", myLane.waypoints)


    def waypoints_cb(self, waypoints):
        self.waypoints_data = waypoints.waypoints;

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

    def dst(self, x1, y1, x2, y2):
    		# Calculate distance between two points
        dist = math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
