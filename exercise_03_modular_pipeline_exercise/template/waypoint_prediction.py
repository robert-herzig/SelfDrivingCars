import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as  the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    i=0
    curvature = 0
    for row in waypoints.T:
        if i >= 2:
            pr = cr
            cr = nr
            nr = row
            vec1 = ((nr[0]-cr[0])**2 + (nr[1]-cr[1])**2)**0.5
            vec2 = ((cr[0]-pr[0])**2 + (cr[1]-pr[1])**2)**0.5
            dot_pr = (nr[0] - cr[0])*(cr[0] - pr[0]) + (nr[1] - cr[1])*(cr[1] - pr[1])
            curvature += (dot_pr / (vec1 * vec2))
        elif i == 0:
            cr = row
            i += 1
        elif i == 1:
            nr = row
            i += 1



    return curvature


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints.reshape(2,-1))**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    t = np.linspace(0, 1, num_waypoints)
    lane_boundary1_points = np.array(splev(t, roadside1_spline))
    lane_boundary2_points = np.array(splev(t, roadside2_spline))
    way_points_center = np.add(lane_boundary1_points, lane_boundary2_points) / 2

    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points

        # output way_points with shape(2 x Num_waypoints)

        #print(way_points)

        return way_points_center
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments

        # derive roadside points from spline

        # derive center between corresponding roadside points
        
        # optimization


        way_points = minimize(smoothing_objective,
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    cur = curvature(waypoints)
    exp_term = np.exp(- exp_constant * np.absolute(len(waypoints.T) - 2 - cur))



    return (max_speed - offset_speed) * exp_term + offset_speed