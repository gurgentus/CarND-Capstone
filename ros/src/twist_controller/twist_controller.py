import rospy
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self):
        # TODO: Implement
	self.speed_pid = PID(100.0, 0.0, 0)
	self.steer_pid = PID(0.2, 0.005, 10.0)
        pass

    def reset():
	self.speed_pid.reset()
	self.steer_pid.reset()

    def control(self, cur_speed, target_speed, cur_angle, target_angle):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
	breaking = 0.0
	throttle = self.speed_pid.step(target_speed-cur_speed, 0.02)

	if (throttle < 0):
	  breaking = abs(throttle)

	if (target_angle > 0):
	  steering = 1.0
	else:
	  steering = -1.0

	steering = self.steer_pid.step(target_angle, 0.02)

	#steering = min(-steering*3.14/6, -50*target_angle)

        return throttle, breaking, steering
