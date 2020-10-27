import os
import json
import random
import numpy as np
from gym import spaces

from ENV.render import render


class Pose_Env_Base:
    def __init__(self, config_path="PoseEnvLarge_multi.json", render=False, render_save=False):
        self.ENV_PATH = 'ENV'

        self.SETTING_PATH = os.path.join(self.ENV_PATH, config_path)
        with open(self.SETTING_PATH, encoding='utf-8') as f:
            setting = json.load(f)

        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.n = len(self.cam_id)
        self.discrete_actions = setting['discrete_actions']
        self.cam_area = np.array(setting['cam_area'])

        self.num_target = setting['target_num']
        self.continous_actions_player = setting['continous_actions_player']
        self.reset_area = setting['reset_area']

        self.max_steps = setting['max_steps']
        self.visual_distance = setting['visual_distance']
        self.safe_start = setting['safe_start']
        self.start_area = self.get_start_area(self.safe_start[0], self.visual_distance)

        # define action space
        self.action_space = [spaces.Discrete(len(self.discrete_actions)) for i in range(self.n)]
        self.rotation_scale = setting['rotation_scale']

        # define observation space
        self.state_dim = 4  # related with the preprocess
        self.observation_space = np.zeros((self.n, self.num_target, self.state_dim), int)

        self.render = render
        self.render_save = render_save

        self.cam = dict()
        for i in range(len(self.cam_id) + 1):
            self.cam[i] = dict(
                location=[0, 0],
                rotation=[0],
            )

        self.count_steps = 0

        # construct target_agent
        self.random_agents = [GoalNavAgent(i, self.continous_actions_player, self.reset_area)
                              for i in range(self.num_target)]

    def set_location(self, cam_id, loc):
        self.cam[cam_id]['location'] = loc

    def get_location(self, cam_id):
        return self.cam[cam_id]['location']

    def set_rotation(self, cam_id, rot):
        for i in range(len(rot)):
            if rot[i] > 180:
                rot[i] -= 360
            if rot[i] < -180:
                rot[i] += 360
        self.cam[cam_id]['rotation'] = rot

    def get_rotation(self, cam_id):
        return self.cam[cam_id]['rotation']

    def get_hori_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt) / np.pi * 180 - current_pose[2]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def get_distance(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        d = np.sqrt(y_delt * y_delt + x_delt * x_delt)
        return d

    def reset(self):

        # reset targets
        self.target_pos_list = np.array([[
            float(np.random.randint(self.start_area[0], self.start_area[1])),
            float(np.random.randint(self.start_area[2], self.start_area[3]))] for _ in range(self.num_target)])
        # reset agent
        for i in range(len(self.random_agents)):
            self.random_agents[i].reset()

        # reset camera
        camera_id_list = [i for i in self.cam_id]
        random.shuffle(camera_id_list)

        for i, cam in enumerate(self.cam_id):
            cam_loc = [np.random.randint(self.cam_area[i][0], self.cam_area[i][1]),
                       np.random.randint(self.cam_area[i][2], self.cam_area[i][3])
                       ]
            self.set_location(camera_id_list[i], cam_loc)  # shuffle

        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)
            cam_rot[0] = np.random.rand()*360
            self.set_rotation(cam, cam_rot)


        self.count_steps = 0

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps
        )

        gt_directions = []
        gt_distance = []
        cam_info = []
        for i, cam in enumerate(self.cam_id):
            # for target navigation
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            cam_info.append([cam_loc, cam_rot])
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(cam_loc + cam_rot)

        info['Directions'] = np.array(gt_directions)
        info['Distance'] = np.array(gt_distance)
        info['Target_Pose'] = np.array(self.target_pos_list)
        info['Reward'], info['Global_reward'] = self.multi_reward(cam_info)

        state, self.state_dim = self.preprocess_pose(info)
        return state

    def step(self, actions):

        info = dict(
            Done=False,
            Reward=[0 for i in range(self.n)],
            Target_Pose=[],
            Cam_Pose=[],
            Steps=self.count_steps
        )

        actions = np.squeeze(actions)

        # actions for cameras
        actions2cam = []
        for i in range(self.n):
            actions2cam.append(self.discrete_actions[actions[i]])

        # target move
        delta_time = 0.3
        for i in range(self.num_target):
            loc = list(self.target_pos_list[i])
            action = self.random_agents[i].act(loc)

            target_hpr_now = np.array(action[1:])
            delta_x = target_hpr_now[0] * action[0] * delta_time
            delta_y = target_hpr_now[1] * action[0] * delta_time
            while loc[0] + delta_x < self.reset_area[0] or loc[0] + delta_x > self.reset_area[1] or \
                    loc[1] + delta_y < self.reset_area[2] or loc[1] + delta_y > self.reset_area[3]:
                action = self.random_agents[i].act(loc)

                target_hpr_now = np.array(action[1:])
                delta_x = target_hpr_now[0] * action[0] * delta_time
                delta_y = target_hpr_now[1] * action[0] * delta_time

            self.target_pos_list[i][0] += delta_x
            self.target_pos_list[i][1] += delta_y

        # camera move
        for i, cam in enumerate(self.cam_id):
            cam_rot = self.get_rotation(cam)
            cam_rot[0] += actions2cam[i][0] * self.rotation_scale
            self.set_rotation(cam, cam_rot)

        cam_info = []
        for i, cam in enumerate(self.cam_id):
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            cam_info.append([cam_loc, cam_rot])

        # r: every camera complete its goal; [camera_num]
        # gr: coverage rate; [1]
        r, gr = self.multi_reward(cam_info)
        # cost by rotation
        cost = 0
        for i, cam in enumerate(self.cam_id):
            if actions[i] != 0:
                r[i] += -0.01
                cost += 1

        info['cost'] = cost/self.n
        info['Reward'] = np.array(r)
        info['Global_reward'] = np.array(gr)

        gt_directions = []
        gt_distance = []
        for i, cam in enumerate(self.cam_id):
            # for target navigation
            cam_loc = self.get_location(cam)
            cam_rot = self.get_rotation(cam)
            gt_directions.append([])
            gt_distance.append([])
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                gt_directions[i].append([angle_h])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                gt_distance[i].append(d)

            info['Cam_Pose'].append(self.get_location(cam) + self.get_rotation(cam))

        info['Target_Pose'] = np.array(self.target_pos_list)
        info['Distance'] = np.array(gt_distance)
        info['Directions'] = np.array(gt_directions)

        # Target_mutual_distance
        gt_target_mu_distance = np.zeros([self.num_target, self.num_target])
        for i in range(self.num_target):
            for j in range(i + 1):
                d = self.get_distance(self.target_pos_list[i], self.target_pos_list[j])
                gt_target_mu_distance[i, j] = d
                gt_target_mu_distance[j, i] = d
        info['Target_mutual_distance'] = gt_target_mu_distance

        self.count_steps += 1

        # set your done condition
        if self.count_steps > self.max_steps:
            info['Done'] = True

        reward = info['Global_reward']

        # show
        if self.render:
            render(info['Cam_Pose'], np.array(self.target_pos_list), gr, save=self.render_save)

        state, self.state_dim = self.preprocess_pose(info)
        return state, reward, info['Done'], info

    def close(self):
        pass

    def seed(self, para):
        pass

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0] - safe_range, safe_start[0] + safe_range,
                      safe_start[1] - safe_range, safe_start[1] + safe_range]
        return start_area

    def angle_reward(self, angle_h, d):
        hori_reward = 1 - abs(angle_h) / 45.0
        visible = hori_reward > 0 and d <= self.visual_distance
        if visible:
            reward = np.clip(hori_reward, -1, 1)
        else:
            reward = -1
        return reward, visible

    def multi_reward(self, cam_info):
        # generate reward
        camera_local_rewards = []
        coverage_rate = []
        for i, cam in enumerate(self.cam_id):
            cam_loc, cam_rot = cam_info[i]
            local_rewards = []
            for j in range(self.num_target):
                angle_h = self.get_hori_direction(cam_loc + cam_rot, self.target_pos_list[j])
                d = self.get_distance(cam_loc + cam_rot, self.target_pos_list[j])
                reward, visible = self.angle_reward(angle_h, d)
                if visible:
                    coverage_rate.append(j)
                    local_rewards.append(reward)
            camera_local_rewards.append(np.mean(local_rewards) if len(local_rewards) > 0 else 0)

        # real coverage rate
        if len(set(coverage_rate)) == 0:
            coverage_rate = [-0.1]
        else:
            coverage_rate = [len(set(coverage_rate)) / self.num_target]

        return camera_local_rewards, coverage_rate

    def preprocess_pose(self, info):
        cam_pose_info = np.array(info['Cam_Pose'])
        target_pose_info = np.array(info['Target_Pose'])
        angles = info['Directions']
        distances = info['Distance']

        camera_num = len(cam_pose_info)
        target_num = len(target_pose_info)

        # normalize center
        center = np.mean(cam_pose_info[:, :2], axis=0)
        cam_pose_info[:, :2] -= center
        if target_pose_info is not None:
            target_pose_info[:, :2] -= center

        # scale
        norm_d = int(max(np.linalg.norm(cam_pose_info[:, :2], axis=1, ord=2))) + 1e-8
        cam_pose_info[:, :2] /= norm_d
        if target_pose_info is not None:
            target_pose_info[:, :2] /= norm_d

        state_dim = 4  # related with the following definition
        feature_dim = target_num * state_dim
        state = np.zeros((camera_num, feature_dim))
        for cam_i in range(camera_num):
            # target info
            target_info = []
            for target_j in range(target_num):
                [angle_h] = angles[cam_i, target_j]
                target_angle = [cam_i / camera_num, target_j / target_num, angle_h / 180]
                line = target_angle + [distances[cam_i, target_j] / 2000]  # 2000 is related with the area of cameras
                target_info += line
            target_info = target_info + [0] * (feature_dim - len(target_info))
            state[cam_i] = target_info
        state = state.reshape((camera_num, target_num, state_dim))
        return state, state_dim


class GoalNavAgent(object):

    def __init__(self, id, action_space, goal_area, goal_list=None):
        self.id = id
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.velocity_high = action_space['high'][0]
        self.velocity_low = action_space['low'][0]
        self.angle_high = action_space['high'][1]
        self.angle_low = action_space['low'][1]
        self.goal_area = goal_area
        self.goal_list = goal_list
        self.goal = self.generate_goal(self.goal_area)

        self.max_len = 100

    def act(self, pose):
        self.step_counter += 1
        if len(self.pose_last[0]) == 0:
            self.pose_last[0] = np.array(pose)
            self.pose_last[1] = np.array(pose)
            d_moved = 30
        else:
            d_moved = min(np.linalg.norm(np.array(self.pose_last[0]) - np.array(pose)),
                          np.linalg.norm(np.array(self.pose_last[1]) - np.array(pose)))
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(pose)
        if self.check_reach(self.goal, pose) or d_moved < 10 or self.step_counter > self.max_len:
            self.goal = self.generate_goal(self.goal_area)
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)

            self.step_counter = 0

        delt_unit = (self.goal[:2] - pose[:2]) / np.linalg.norm(self.goal[:2] - pose[:2])
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [velocity, delt_unit[0], delt_unit[1]]

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.goal_id = 0
        self.goal = self.generate_goal(self.goal_area)
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]

    def generate_goal(self, goal_area):
        if self.goal_list and len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            x = np.random.randint(goal_area[0], goal_area[1])
            y = np.random.randint(goal_area[2], goal_area[3])
            goal = np.array([x, y])
        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 5
