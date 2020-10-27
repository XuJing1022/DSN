import os
import argparse
import random
import time
import numpy as np

from ENV.pose_env_base import Pose_Env_Base

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='DSN')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--sleep-time', type=int, default=0, metavar='LO', help='seconds')
parser.add_argument('--test-eps', type=int, default=1, metavar='M', help='testing episode length')
parser.add_argument('--max-step', type=int, default=100, metavar='LO', help='max learning steps')
parser.add_argument('--render', dest='render', action='store_true', help='render')
parser.add_argument('--save', dest='save', action='store_true', help='render save')


class RandomAgent:
    def __init__(self, obs_space, action_space):
        self.action_dim = action_space[0].n

    def act(self, obs):
        return random.randint(0, self.action_dim - 1)


def test():
    args = parser.parse_args()
    env = Pose_Env_Base(render=args.render, render_save=args.save)
    env.seed(args.seed)
    start_time = time.time()
    count_eps = 0

    rand_agent = RandomAgent(env.observation_space, env.action_space)

    while True:
        AG = 0
        reward_sum = np.zeros(1)  # team reward
        reward_sum_list = []
        len_sum = 0
        for i_episode in range(args.test_eps):
            state = env.reset()
            reward_sum_ep = np.zeros(1)
            rotation_sum_ep = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []
            while True:
                actions = []
                for i in range(len(state)):
                    actions.append(rand_agent.act(state[i]))
                state_multi, reward, done, info = env.step(actions)
                len_sum += 1
                fps_counter += 1
                reward_sum_ep += reward
                rotation_sum_ep += info['cost']
                if done:
                    AG += reward_sum_ep / rotation_sum_ep
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep)
                    fps = fps_counter / (time.time() - t0)
                    fps_all.append(fps)
                    break

        # player.max_length:
        ave_AG = AG / args.test_eps
        ave_reward_sum = reward_sum / args.test_eps
        len_mean = len_sum / args.test_eps
        reward_step = reward_sum / len_sum
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)

        print("Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
              "mean reward {5}, std reward {6}, AG {7}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)))


if __name__ == '__main__':
    test()
