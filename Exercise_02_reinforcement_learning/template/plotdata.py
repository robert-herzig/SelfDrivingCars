from utils import get_state, visualize_training



with open('reward.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
episode_rewards = [x.strip() for x in content]

episode_rewards_f=[float(i) for i in episode_rewards]

with open('training.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
training_losses = [x.strip() for x in content]

training_losses_f=[float(i) for i in training_losses]
print( 'done')

visualize_training(episode_rewards_f, training_losses_f, 'agent_ts200000,bs12000,ef0.75,exp0.02,ar4,bat32,gm0.999')

print( 'done2')