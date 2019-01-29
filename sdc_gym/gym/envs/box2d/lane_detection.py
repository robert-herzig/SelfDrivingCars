import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def load_crop_gray(path, cropsize=68):
    state_image_full = np.load(path)
    
    # check channels channels
    assert state_image_full.shape[2] == 3, "wrong shape...should be 96x96x3"
    state_image = state_image_full[:cropsize,:][::-1]
    
    # to grayscale
    gray_state = (state_image[:,:,0] + state_image[:,:,1] + state_image[:,:,2])/3
    
    return gray_state, state_image, state_image_full[::-1]


def crop_gray(state_image_full, cropsize=68):
    # check channels channels
    assert state_image_full.shape[2] == 3, "wrong shape...should be 96x96x3"
    state_image = state_image_full[:cropsize,:][::-1]
    
    # to grayscale
    gray_state = (state_image[:,:,0] + state_image[:,:,1] + state_image[:,:,2])/3
    
    return gray_state, state_image, state_image_full[::-1]


def edge_detection(gray_image, threshold=14):
    # derive of gradient
    gradient = abs(np.gradient(gray_image)[0]) + abs(np.gradient(gray_image)[1]) 
    
    # thresholding
    upper_threshold_indices = gradient < threshold
    gradient[upper_threshold_indices] = 0
    
    return gradient


def lane_detection(gradient, tck1_old, tck2_old, self_position=np.array([48,0]), self_position_label=1, spline_smoothness=10):
    x_maxima = []
    y_maxima = []

    for i in range(1,gradient.shape[0]):
        argmaxima = find_peaks(gradient[i],distance=3)[0]
        for ii in range(argmaxima.shape[0]):
            x_maxima.append([argmaxima[ii]])
            y_maxima.append([i])


    argmaxima = find_peaks(gradient[0],distance=8)[0]
    argmaxima = argmaxima[argmaxima > 1]
    argmaxima = argmaxima[argmaxima < 95]
    #assert argmaxima.shape[0]==2

    maxima = np.concatenate((np.array(x_maxima), np.array(y_maxima)), axis=1)
    # lane1 = np.array([[argmaxima[argmaxima - self_position[0]<=0][0],  0]])
    A = np.argsort((argmaxima - self_position[0])**2)
    if argmaxima.shape[0]>=2:
        if argmaxima[A[0]] > argmaxima[A[1]]:
            lane1 = np.array([[argmaxima[A[1]],  0]])
            lane2 = np.array([[argmaxima[A[0]],  0]])
        else:
            lane2 = np.array([[argmaxima[A[1]],  0]])
            lane1 = np.array([[argmaxima[A[0]],  0]])

        run = True
        while run:
            closest = np.argmin(np.sum((maxima - lane1[-1])**2, axis=1))

            if np.sum((maxima[closest] - lane1[-1])**2) >=100 or maxima.shape[0]==1:
                run = False
            else:
                lane1 = np.append(lane1, list([maxima[closest]]), axis=0)
                maxima = np.delete(maxima, closest, axis=0)
        #maxima = np.concatenate((np.array(x_maxima), np.array(y_maxima)), axis=1)
        # lane2 = np.array([[argmaxima[argmaxima - self_position[0]>=0][0],  0]])
        run = True
        while run:
            closest = np.argmin(np.sum((maxima - lane2[-1])**2, axis=1))
            if np.sum((maxima[closest] - lane2[-1])**2) >=100 or maxima.shape[0]==1:
                run = False
            else:
                lane2 = np.append(lane2, list([maxima[closest]]), axis=0)
                maxima = np.delete(maxima, closest, axis=0)
            
        if lane1.shape[0] > 20 and lane2.shape[0] > 20:
            #print(lane1.shape[0], lane2.shape[0])
            tck1, u1 = splprep([lane1[0:,0], lane1[0:,1]], s=spline_smoothness, k=2)
            tck2, u2 = splprep([lane2[0:,0], lane2[0:,1]], s=spline_smoothness, k=2)
        else:
            tck1 = tck1_old
            tck2 = tck2_old
    
    else:
        tck1 = tck1_old
        tck2 = tck2_old
    
    return tck1, tck2


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])


def smoothing_objective(way_points, way_points_center, weight_curvature=-20):
    ls_tocenter = np.mean((way_points_center - way_points.reshape(2,-1))**2)
    norm_diff = normalize(np.diff(way_points.reshape(2,-1), axis=1))
    curvature = np.sum(norm_diff[:,:-1] * norm_diff[:,1:])
    return weight_curvature * curvature + ls_tocenter


def con(t):
    return t.reshape(2,-1)[:,0] - np.array([47,0]) 


cons = {'type':'eq', 'fun': con}


def waypoint_prediction(tck11, tck22, num_waypoints=20, parameter_bound_waypoints=1, way_type = "smoothed"):
    if way_type == "center":
        t = np.linspace(0, parameter_bound_waypoints, num_waypoints)
        lane1_points = np.array(splev(t, tck11))
        lane2_points = np.array(splev(t, tck22))
        way_points = np.array(lane1_points + lane2_points)/2
        return way_points+0.5, lane1_points+0.5, lane2_points+0.5
    
    elif way_type == "smoothed":
        t = np.linspace(0, parameter_bound_waypoints, num_waypoints)
        lane1_points = np.array(splev(t, tck11))
        lane2_points = np.array(splev(t, tck22))
        way_points_center = np.array(lane1_points + lane2_points)/2
        way_points = np.array(lane1_points + lane2_points)/2
        
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)+0.5, lane1_points+0.5, lane2_points+0.5
    
    
def lateral_control(former_steering_angle, car_position, 
                    car_velocity, car_angle, 
                    waypoints, gain_constant, 
                    damping_constant):
    if False:
        steering_angle = np.arctan((waypoints[0,1]-waypoints[0,0])/(waypoints[1,1]-waypoints[1,0]+0.00001))
        damping = -1*damping_constant*(steering_angle - former_steering_angle)
        print(waypoints[:,0])
        print(48 - waypoints[0,0])
        return np.clip(steering_angle + damping, -0.4, 0.4)/0.4 
    
    if True:
        way_angle = np.arctan((waypoints[0,1]-waypoints[0,0])/(waypoints[1,1]-waypoints[1,0]+0.00001))
        steering_angle = way_angle + 0.5*np.arctan(-5*(47.5 - waypoints[0,0])/(car_velocity + 0.00001))
        damping = -1*damping_constant*(steering_angle - former_steering_angle)
        return np.clip(steering_angle + damping, -0.4, 0.4)/0.4 

# TODO change
class PID:
	"""
	Discrete PID control
	"""

	def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500):

		self.Kp=P
		self.Ki=I
		self.Kd=D
		self.Derivator=Derivator
		self.Integrator=Integrator
		self.Integrator_max=Integrator_max
		self.Integrator_min=Integrator_min

		self.set_point=0.0
		self.error=0.0

	def update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""

		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		PID = self.P_value + self.I_value + self.D_value

		return PID

def PID_update(current_value, target, former_error, sum_error, KP=0.006, KI=0.0, KD=0.006, Derivator=0, Integrator=0, Integrator_max=500, Integrator_min=-500):
    error = target - current_value
    P_value = KP * error
    D_value = KD * ( error - former_error ) 

    sum_error += error
    I_value = KI * sum_error
    control = P_value + I_value + D_value
    return control, error, sum_error

def longitut_control(speed, waypoints, former_error, sum_error):
    target_speed = predict_target_speed(waypoints)
    control, error, sum_error = PID_update(speed, target_speed, former_error, sum_error)
    brake = 0
    gas = 0
    print(control)
    if control >= 0:
        gas = np.clip(control, 0, 1) 
    else:
        brake = np.clip(-1*control, 0.3, 0.6)

    return gas, brake, error, sum_error


def predict_target_speed(waypoints):
    norm_diff = normalize(np.diff(waypoints, axis=1))[:,:4]
    curvature = abs(norm_diff.shape[1]-1 - np.sum(norm_diff[:,:-1] * norm_diff[:,1:]))
    if curvature <= 5:
        return 150 * np.exp(-4.5*curvature)
    else:
        return 60 

def plot_state_lane_wayp(lane1_points, lane2_points, way_points, state_cropped, steps):
    
    fig = plt.figure()

    plt.plot(lane1_points[0], lane1_points[1]+96-68, linewidth=5, color='orange')
    plt.plot(lane2_points[0], lane2_points[1]+96-68, linewidth=5, color='orange')
    plt.scatter(way_points[0], way_points[1]+96-68, color='white')
    #plt.scatter(48-0.5, 24-0.5, color='black')
    plt.imshow(state_cropped)
    plt.axis('off')
    plt.xlim((-0.5,95.5))
    plt.ylim((-0.5,95.5))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig("../car_videos/"+str(steps).zfill(4)+"test.jpeg", bbox_inches = 'tight', pad_inches = 0)

def extract_sensor_values(observation, batch_size):
    """
    observation:    python list of batch_size many torch.Tensors of size
                    (96, 96, 3)
    batch_size:     int
    return          torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 4),
                    torch.Tensors of size (batch_size, 1),
                    torch.Tensors of size (batch_size, 1)
    """
    speed_crop = observation[84:94, 12, 0]
    speed = speed_crop.sum() / 255
    # abs_crop = observation[84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    # abs_sensors = abs_crop.sum(dim=1) / 255
    # steer_crop = observation[88, 38:58, 1].reshape(batch_size, -1)
    # steering = steer_crop.sum(dim=1, keepdim=True)
    # gyro_crop = observation[88, 58:86, 0].reshape(batch_size, -1)1
    # gyroscope = gyro_crop.sum(dim=1, keepdim=True)
    return speed #, abs_sensors.reshape(batch_size, 4), steering, gyroscope