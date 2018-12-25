import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=3, damping_constant=1):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.damping_constant2 = 0.1
        self.previous_steering_angle = 0
        self.previous_cross_track_error = 0

    def unit_vector(self,vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'

        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position

        # derive stanley control law
        # prevent division by zero by adding as small epsilon

        # derive damping term
        '''print(waypoints[0][0],waypoints[1][0])
        print(waypoints[0][1],waypoints[1][1])
        desired_vector_length = ((waypoints[0][0] - waypoints [1][0])**2 + (waypoints[1][0] - waypoints[1][1])**2)**0.5
        desired_vector = [(waypoints[0][0] - waypoints [1][0])/desired_vector_length,(waypoints[1][0] - waypoints[1][1])/desired_vector_length]
        print(desired_vector)
        orientation_error = np.arcsin(np.cross([1,0],desired_vector))
        print(orientation_error)'''

        desired_vector_length = ((waypoints[0][0] - waypoints[0][1]) ** 2 + (
        waypoints[1][0] - waypoints[1][1]) ** 2) ** 0.5
        desired_vector = [-(waypoints[1][0] - waypoints[1][1]) / desired_vector_length,
                          -(waypoints[0][0] - waypoints[0][1]) / desired_vector_length]
        #print(desired_vector)

        orientation_error = desired_vector[1]
        #print(orientation_error)


        cross_track_error = waypoints[0][0] - 48
        diff_cross_track_error = -(cross_track_error - self.previous_cross_track_error)

        steering_angle = orientation_error+ np.arctan((self.gain_constant*cross_track_error + self.damping_constant * diff_cross_track_error) / speed + 0.01)
        #print(steering_angle)

        steering_angle = steering_angle - self.damping_constant2 * (steering_angle - self.previous_steering_angle)
        self.previous_steering_angle = steering_angle
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






