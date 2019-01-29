import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step DONE
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
<<<<<<< HEAD
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
    rew_tensor = torch.from_numpy(rew_batch).to("cuda")
    rew_prediction = policy_net(obs_batch)
    # rew_prediction=rew_prediction.detach().cpu().numpy()
    next_rew_prediction = target_net(next_obs_batch)
    # print(next_rew_prediction)
    max_idx = torch.argmax(next_rew_prediction, 1)
    # print(max_idx)
    # next_rew_prediction = next_rew_prediction.detach().cpu().numpy()
    # print(rew_prediction[np.arange(batch_size),act_batch])
    # print(rew_batch.shape)
    # print(next_rew_prediction[np.arange(batch_size),max_idx])
    # print(rew_prediction[np.arange(batch_size),act_batch])
    loss = torch.mean(torch.pow(
        rew_tensor + gamma * next_rew_prediction[np.arange(batch_size), max_idx] - rew_prediction[
            np.arange(batch_size), act_batch], 2))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss)
=======

    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
    rew_tensor = torch.from_numpy(rew_batch).to(device)
    rew_prediction = policy_net(obs_batch)

    next_rew_prediction = target_net(next_obs_batch) #here target net is used to estimate the value (Double DQN)
                                                     #for normal DQN switch to policy net

    max_idx = torch.argmax(next_rew_prediction,1)     #select maximum best action

    loss = torch.mean( torch.pow( rew_tensor+ gamma* next_rew_prediction[np.arange(batch_size),max_idx]\
                                - rew_prediction[np.arange(batch_size),act_batch],2))                     #calculate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

>>>>>>> 040f97d055cef251b35e7e8ebf4b73a8d1f476ca
    return loss


def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network DONE
    target_net.load_state_dict(policy_net.state_dict())
