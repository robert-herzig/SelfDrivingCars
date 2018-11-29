import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size
        # TODO: Create network  #DONE
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=3, padding=2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.sensor_net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(True),
            nn.Linear(128,128)
        )
        self.classification = nn.Sequential(
            nn.Linear(32 * 4 * 4 + 128, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, action_size)
        )



    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network #DONE
        observation=torch.Tensor(observation).to(self.device)
        #print(observation.shape)
        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(observation, observation.shape[0])
        sensor_inputs = torch.cat((speed, abs_sensors, steering, gyroscope), 1)
        image_input = observation.permute(0, 3, 1, 2)

        conv_outputs = self.convolutions(image_input)
        conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)

        sensor_outputs = self.sensor_net(sensor_inputs)

        combined_output = torch.cat((sensor_outputs, conv_outputs), 1)

        final_output = self.classification(combined_output)

        return final_output



    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
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