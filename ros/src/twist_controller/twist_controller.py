import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self):
        # TODO: Implement
        pass

    def control(self, cur_speed, target_speed, cur_angle, target_angle):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
	if (cur_speed < target_speed):
	  throttle = 1.0
	else:
	  throttle = -1.0

	if (target_angle > 0):
	  steering = 0.4
	else:
	  steering = -0.4
	steering = 10*target_angle

        return throttle, 0., steering
