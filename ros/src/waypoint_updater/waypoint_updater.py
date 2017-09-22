#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import tf
from std_msgs.msg import Int32

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

LOOKAHEAD_WPS = 400 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Subscribers for /traffic_waypoint and /obstacle_waypoint added
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # Publish final waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Variables
        self.pose = None
        self.waypoints = None

        # Loop
        self.loop()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

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

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_final_waypoints()
            rate.sleep()
        
    def publish_final_waypoints(self):
        if self.waypoints != None and self.pose != None:

            time = rospy.Time().now()
            current_position = self.pose.pose.position
            waypoints = self.waypoints.waypoints
            velocity = 20

            # Find nearest waypoint
            closest_waypoint_index = None
            closest_waypoint_distance = float('inf')
            for i in range(len(waypoints)):
                waypoint_position = waypoints[i].pose.pose.position
                distance = math.sqrt((current_position.x-waypoint_position.x)**2 + (current_position.y-waypoint_position.y)**2 + (current_position.z-waypoint_position.z)**2)
                if distance < closest_waypoint_distance:
                    closest_waypoint_index = i
                    closest_waypoint_distance = distance

            # Increase waypoint index if it's behind current position
            delta_py = waypoints[closest_waypoint_index].pose.pose.position.y - current_position.y
            delta_px = waypoints[closest_waypoint_index].pose.pose.position.y - current_position.x
            heading  = math.atan2(delta_py, delta_px)
            euler_angles_xyz = tf.transformations.euler_from_quaternion([self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w])
            theta = euler_angles_xyz[-1]
            angle = math.fabs(theta-heading)
            if angle > math.pi/4:
                closest_waypoint_index += 1

            # Publish lane
            lane = Lane()
            lane.header.stamp = time
            lane.header.frame_id = '/world'
            for i in range(closest_waypoint_index, closest_waypoint_index + LOOKAHEAD_WPS):
                index = i % len(waypoints)
                waypoint = waypoints[index]
                waypoint.twist.twist.linear.x = velocity
                lane.waypoints.append(waypoint)
            self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
