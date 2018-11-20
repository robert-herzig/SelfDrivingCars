import os
import numpy as np
import gym
from pyglet.window import key


def load_imitations(data_folder):
    """
    1.1 a)
    Given the folder containing the expert imitations, the data gets loaded and
    stored it in two lists: observations and actions.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    return:
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    files = os.listdir(data_folder)
    observations = []
    actions = []

    for file in files:
        full_array = np.load(os.path.join(data_folder, file))
        if "observation" in str(file):
            # full_array = full_array.reshape(3, 96, 96)
            # print("OBS SHAPE = " + str(full_array.shape))
            observations.append(full_array)
        else:
            actions.append(full_array)

    observations = np.array(observations)
    actions = np.array(actions)

    print(observations.shape)
    print(actions.shape)

    return observations, actions




def save_imitations(data_folder, actions, observations):
    """
    1.1 f)
    Save the lists actions and observations in numpy .npy files that can be read
    by the function load_imitations.
                    N = number of (observation, action) - pairs
    data_folder:    python string, the path to the folder containing the
                    observation_%05d.npy and action_%05d.npy files
    observations:   python list of N numpy.ndarrays of size (96, 96, 3)
    actions:        python list of N numpy.ndarrays of size 3
    """
    n = 0
    while os.path.exists(data_folder + '/action_%05d.npy' % n):
        n = n + 1

    for action, observation in zip(actions, observations):
        imitations_folder = os.path.join(data_folder, 'action_%05d.npy' % n)
        np.save(imitations_folder, action)
        imitations_folder = os.path.join(data_folder, 'observation_%05d.npy' % n)
        np.save(imitations_folder, observation)
        n += 1

class ControlStatus:
    """
    Class to keep track of key presses while recording imitations.
    """

    def __init__(self):
        self.stop = False
        self.save = False
        self.quit = False
        self.steer = 0.0
        self.accelerate = 0.0
        self.brake = 0.0

    def key_press(self, k, mod):
        if k == key.ESCAPE: self.quit = True
        if k == key.SPACE: self.stop = True
        if k == key.TAB: self.save = True
        if k == key.LEFT: self.steer = -1.0
        if k == key.RIGHT: self.steer = +1.0
        if k == key.UP: self.accelerate = +0.5
        if k == key.DOWN: self.brake = +0.8

    def key_release(self, k, mod):
        if k == key.LEFT and self.steer < 0.0: self.steer = 0.0
        if k == key.RIGHT and self.steer > 0.0: self.steer = 0.0
        if k == key.UP: self.accelerate = 0.0
        if k == key.DOWN: self.brake = 0.0


def record_imitations(imitations_folder):
    """
    Function to record own imitations by driving the car in the gym car-racing
    environment.
    imitations_folder:  python string, the path to where the recorded imitations
                        are to be saved

    The controls are:
    arrow keys:         control the car; steer left, steer right, gas, brake
    ESC:                quit and close
    SPACE:              restart on a new track
    TAB:                save the current run
    """
    env = gym.make('CarRacing-v0').env
    status = ControlStatus()
    total_reward = 0.0

    while not status.quit:
        observations = []
        actions = []
        # get an observation from the environment
        observation = env.reset()
        env.render()

        # set the functions to be called on key press and key release
        env.viewer.window.on_key_press = status.key_press
        env.viewer.window.on_key_release = status.key_release

        while not status.stop and not status.save and not status.quit:
            # collect all observations and actions
            observations.append(observation.copy())
            actions.append(np.array([status.steer, status.accelerate,
                                     status.brake]))
            # submit the users' action to the environment and get the reward
            # for that step as well as the new observation (status)
            observation, reward, done, info = env.step([status.steer,
                                                        status.accelerate,
                                                        status.brake])
            total_reward += reward
            env.render()

        if status.save:
            print("Save Imitations")
            save_imitations(imitations_folder, actions, observations)
            status.save = False

        status.stop = False
        env.close()

# if __name__ == '__main__':
#     load_imitations("./data/teacher")
