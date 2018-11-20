import torch
import random
import time
from network import ClassificationNetwork
from imitations import load_imitations
import numpy as np


def train(data_folder, trained_network_file):
    """
    Function for training the network.
    """
    gpu = torch.device('cuda')
    infer_action = ClassificationNetwork()
    infer_action.to(gpu)   #make the network run on the gpu
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    observations, actions = load_imitations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]
    

    nr_epochs = 100
    batch_size = 20
    number_of_classes = 9  # needs to be changed
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(gpu))
            batch_gt.append(batch[1].to(gpu))
            
            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0),
                                         (-1, number_of_classes))
                dim = list(batch_in.shape)

                speed, abs_sensors, steering, gyroscope = infer_action.extract_sensor_values(batch_in,dim[0])
               
                sensors = torch.cat((speed, abs_sensors, steering, gyroscope),1)

                batch_out = infer_action(batch_in.permute(0,3,1,2),sensors)    #changed the order of dimensions
                loss = cross_entropy_loss(batch_out, batch_gt.float())  #targets can only be long
                                     

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss


                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

        

    torch.save(infer_action, trained_network_file)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
    batch_out:      torch.Tensor of size (batch_size, number_of_classes)
    batch_gt:       torch.Tensor of size (batch_size, number_of_classes)
    return          float
    """
    a=batch_out*batch_gt
    a=torch.clamp(a,min=1e-12,max=1-1e-12)
    cross,_=torch.max(a,1)
    #print(a)
    output = - cross.log().sum()
    

    return output
