import gym
import deepq
import sys





if __name__ == '__main__':
    env = gym.make("CarRacing-v0")
    if len(sys.argv) == 1 :
        deepq.learn(env)
    else:
        deepq.learn(env,float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]), \
                    float(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]),int(sys.argv[9]), \
                    float(sys.argv[10]),int(sys.argv[11]),sys.argv[12])
    env.close()
