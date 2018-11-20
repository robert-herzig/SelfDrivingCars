import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, use_sensors = False, use_multi_binary = False):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        print("Create network with " + str(use_sensors) + "," + str(use_multi_binary))

        self.use_sensors = use_sensors
        self.use_multi_binary = use_multi_binary
                
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )

        self.sensor_net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(True),
            nn.Linear(128, 128)
        )


        if self.use_sensors:
            if self.use_multi_binary:
                self.classification = nn.Sequential(
                    nn.Linear(32 * 24 * 24 + 128, 512),
                    nn.ReLU(True),
                    nn.Linear(512, 4),
                    nn.Sigmoid()
                )
            else:
                self.classification = nn.Sequential(
                    nn.Linear(32*24*24+128, 512),
                    nn.ReLU(True),
                    nn.Linear(512,9)
                )
        else:
            if self.use_multi_binary:
                self.classification = nn.Sequential(
                    nn.Linear(32 * 24 * 24, 512),
                    nn.ReLU(True),
                    nn.Linear(512, 4),
                    nn.Sigmoid()
                )
            else:
                self.classification = nn.Sequential(
                    nn.Linear(32*24*24, 512),
                    nn.ReLU(True),
                    nn.Linear(512,9)
                )



    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, number_of_classes)
        """

        if self.use_sensors:
            # observation = observation.permute(0, 3, 1, 2)
            # print(observation.size())
            observation_ch_last = observation.permute(0, 2, 3, 1)
            # obs_list = []
            # for tobs in observation_ch_last:
            #     obs_list.append(tobs)
            # print(observation_ch_last.size())
            # obs_list = np.array(obs_list)


            speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation_ch_last, 1)
            sensor_inputs = torch.cat((speed, abs_sensors, steering, gyroscope), 1)
            image_input = observation

            conv_outputs = self.convolutions(image_input)
            conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)

            sensor_outputs = self.sensor_net(sensor_inputs)

            total_outputs = torch.cat((sensor_outputs, conv_outputs), 1)

            if self.use_multi_binary:
                observation = self.classification(total_outputs)
            else:
                observation = F.softmax(self.classification(total_outputs),dim=1)     #calculate the last layers and probabilities

            # print(observation)
            return observation
        else:

            image_input = observation

            conv_outputs = self.convolutions(image_input)
            conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)

            if self.use_multi_binary:
                observation = self.classification(conv_outputs)
            else:
                observation = F.softmax(self.classification(conv_outputs),
                                    dim=1)  # calculate the last layers and probabilities
            # print(observation)
            return observation

    

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
                    if actions_np[1] > actions_mse_lossnp[2]: # gas right
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

    def infer_to_multi_class(self, actions):
        actions_list = []

        for action in actions:
            actions_np = action.numpy()
            print(actions_np)
            print("SHAPE: " + str(actions_np.shape))

            try:
                # Input is {steer, gas, break}
                steer_left = 0
                steer_right = 0
                gas = 0
                brake = 0

                if actions_np[0] > 0.1:
                    steer_right = 1
                elif actions_np[0] < -0.1:
                    steer_left = 1

                if actions_np[1] > 0:
                    gas = 1
                elif actions_np[2] > 0:
                    brake = 1

                output = np.array([steer_right, steer_left, gas, brake])
                output_torch = torch.from_numpy(output)
                # print("OUTPUT: " + str(output))
                actions_list.append(output_torch)
            except: #JUST IN CASE FORMAT GOES CRAZY AGAIN
                actions_list.append(torch.from_numpy(np.zeros(4)))

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
        if self.use_multi_binary:
            # print("SCORES: " + str(scores))
            # print("TODO: Convert scores to actions for multi binary class")
            scores_np = scores.data.numpy()[0]
            # print("SHAPE OF NP SCORES: " + str(scores_np.shape))
            steer_angle = 0
            gas = 0
            brake = 0
            if scores_np[0] > 0.1:
                steer_angle = 1
            elif scores_np[1] > 0.1:
                steer_angle = -1

            if scores_np[2] > 0.1:
                gas = 1
            if scores_np[3] > 0.1:
                brake = 1

            return(steer_angle, gas, brake)


        else:
            print("SCORES without multiclasses")
            dummy, max_idx = torch.max(scores,1) # the output of the network is double not boolean

            if max_idx == 0:        #changed the conditions also
                return (0, 0.5, 0)
            elif max_idx == 1:
                return (0, 0, 1)
            elif max_idx == 2:
                return (-1, 0.05, 0)
            elif max_idx == 2:
                return (1, 0.05, 0)
            elif max_idx == 4:
                return (-1, 0.5, 0)
            elif max_idx == 5:
                return (1, 0.5, 0)
            elif max_idx == 6:
                return (-1, 0, 1)
            elif max_idx == 7:
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

        # abs_crop_b4 = observation[:, 84:94, 18:25:2, 2]
        # print("SHAPE OF abs_crop before reshape: " + str(abs_crop_b4.shape))


        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        # abs_crop = abs_crop_b4
        # print(abs_crop.shape)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope