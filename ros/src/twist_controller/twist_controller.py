import rospy
from std_msgs.msg import Float32
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):

    def __init__(self):

        self.last_time = None

        max_lat_accel = 3.0
        max_steer_angle = 8.0
        steer_ratio = 2.67
        wheel_base = 3.0
        self.brake_deadband = 0.2

        self.pid_control  = PID(5.0, 0.1, 0.02)
        self.pid_steering = PID(15.0, 1.2, 0.1)

        rospy.Subscriber('/kp', Float32, self.kp_cb)
        rospy.Subscriber('/ki', Float32, self.ki_cb)
        rospy.Subscriber('/kd', Float32, self.kd_cb)

        self.lpf_pre  = LowPassFilter(0.2, 0.1)
        self.lpf_post = LowPassFilter(0.4, 0.1)

        self.yaw_control  = YawController(wheel_base=wheel_base, steer_ratio=steer_ratio, min_speed=0.0, max_lat_accel=max_lat_accel, max_steer_angle=max_steer_angle)    


    def control(self, dbw_enabled, twist_cmd, current_velocity):
        tc_l = twist_cmd.twist.linear
        tc_a = twist_cmd.twist.angular

        cv_l = current_velocity.twist.linear
        cv_a = current_velocity.twist.angular

        desired_linear_velocity  = tc_l.x
        desired_angular_velocity = tc_a.z

        current_linear_velocity  = cv_l.x
        current_angular_velocity = cv_a.z        
        
        if dbw_enabled is False:
            self.pid_control.reset()
            self.pid_steering.reset()
           
        if self.last_time != None:
            time = rospy.get_time()
            delta_t = time - self.last_time
            self.last_time = time

            velocity_error = desired_linear_velocity - current_linear_velocity
            control = self.pid_control.step(velocity_error, delta_t)
            throttle = max(0.0, control)
            brake = max(0.0, -control) + self.brake_deadband

            desired_steering = self.yaw_control.get_steering(desired_linear_velocity, desired_angular_velocity, desired_linear_velocity)
            current_steering = self.yaw_control.get_steering(current_linear_velocity, current_angular_velocity, current_linear_velocity)

            steering_error = desired_steering - current_steering
            steering_error = self.lpf_pre.filt(steering_error)

            steering = self.pid_steering.step(steering_error, delta_t)
            steering = self.lpf_post.filt(steering)

            return throttle, brake, steering

        else:
            self.last_time = rospy.get_time()
            return 0.0, 0.0, 0.0


    def kp_cb(self, msg):
        self.pid_steering.kp = msg.data

    def ki_cb(self, msg):
        self.pid_steering.ki = msg.data

    def kd_cb(self, msg):
        self.pid_steering.kd = msg.data

