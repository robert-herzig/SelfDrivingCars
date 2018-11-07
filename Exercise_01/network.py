import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super(ClassificationNetwork, self).__init__()
        gpu = torch.device('cuda')

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 16, 5)
        # an affine operation: y = Wx + b
        # self.fc0 = nn.Linear(96 * 96 * 2, 16 * 5 * 5)
        self.fc1 = nn.Linear(14112, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """

        self.cuda()

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(observation)), (2, 2))
        # # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, )
        # x = F.relu(self.fc0(observation))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are number_of_classes
        different classes, then every action is represented by a
        number_of_classes-dim vector which has exactly one non-zero entry
        (one-hot encoding). That index corresponds to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size number_of_classes

        Classes:
        0: Gas
        1: Brake
        2: Left
        3: Right
        4: Gas Left
        5: Gas Right
        6: Brake Left
        7: Brake Right
        8: Idle
        """

        num_classes = 9
        actions_list = []

        for action in actions:
            output_np = np.zeros(num_classes)
            #Input is {steer, gas, break}
            actions_np = action.numpy()
            output_int = 0

            if actions_np[0] > 0: #right
                if actions_np[1] > 0 or actions_np[2] > 0: #gas or brake
                    if actions_np[1] > actions_np[2]: # gas right
                        output_int = 5
                    else: #brake right
                        output_int = 7#only right
                else:
                    output_int = 3
            elif actions_np[0] < 0: #left
                if actions_np[1] > 0 or actions_np[2] > 0: #gas or brake
                    if actions_np[1] > actions_np[2]: # gas left
                        output_int = 4
                    else: #brake left
                        output_int = 6
                else:
                    output_int = 2#only left
            else: #straight
                if actions_np[1] > 0 or actions_np[2] > 0: #gas or brake
                    if actions_np[1] > actions_np[2]: # gas
                        output_int = 0
                    else: #brake
                        output_int = 1#brake
                else:
                    output_int = 8 #idle

            output_np[output_int] = 1
            output_torch = torch.from_numpy(output_np)
            actions_list.append(output_torch)

        return actions_list


    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
        scores:         python list of torch.Tensors of size number_of_classes
        return          (float, float, float)

        Classes:
        0: Gas
        1: Brake
        2: Left
        3: Right
        4: Gas Left
        5: Gas Right
        6: Brake Left
        7: Brake Right
        8: Idle

        """

        if scores[0] == 1:
            return (0, 1, 0)
        elif scores[1] == 1:
            return (0, 0, 1)
        elif scores[2] == 1:
            return (-1, 0, 0)
        elif scores[3] == 1:
            return (1, 0, 0)
        elif scores[4] == 1:
            return (-1, 1, 0)
        elif scores[5] == 1:
            return (1, 1, 0)
        elif scores[6] == 1:
            return (-1, 0, 1)
        elif scores[7] == 1:
            return (1, 0, 1)
        else: #idle even if input has totally wrong format somehow
            return (0, 0, 0)


    def extract_sensor_values(self, observation, batch_size):
        """
        observation:    python list of batch_size many torch.Tensors of size
                        (96, 96, 3)
        batch_size:     int
        return          torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 4),
                        torch.Tensors of size (batch_size, 1),
                        torch.Tensors of size (batch_size, 1)
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope
