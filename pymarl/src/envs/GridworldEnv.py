import numpy as np
import itertools
import os


class GridworldEnv:
    def __init__(self,seed,map_name,episode_limit=30,input_rows=9, input_cols=12,penalty=True,penalty_amount=2,
                 noise=False, noise_num=1, path=None, stochastic=0., noisy_reward=0.):
        """
        obs 0: 空 1: agent -1: wall 2: Goal
        """

        n_agents = 2
        self.noise = noise
        self.noise_num = noise_num
        self.rows, self.cols = input_rows, input_cols
        self.Goal_shape = 2
        self._episode_steps = 0
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.n_actions = 5
        self.map_name=map_name   ##full
        self.obs_range = 2

        if map_name == 'wall4':
            self.n_subgoals = (self.obs_range * 2) * 4
            self.subgoal_shape = 1
            self.obs_shape = (self.obs_range * 2 + 1) ** 2 + 2 + int(self.noise) * self.noise_num
            self.obs_1 = [[0] * (self.obs_range * 2 + 1) for _ in range(self.obs_range * 2 + 1)]
            self.obs_2 = [[0] * (self.obs_range * 2 + 1) for _ in range(self.obs_range * 2 + 1)]
            self.state_shape = self.obs_shape * 2
        else:
            self.obs_shape = (self.rows + self.cols) * 2 + int(self.noise) * self.noise_num
            self.subgoal_shape = self.obs_shape // 2
            self.state_shape = self.obs_shape * 2
        
        self.subgoal_space = range(self.subgoal_shape)
        self.heat_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        
        self.center = self.cols // 2
        ###larger gridworld
        self.visible_row=[i for i in range(self.rows//2-2,self.rows//2+3)]
        self.visible_col=[i for i in range(self.cols//2-3,self.cols//2+3)]
        self.vision_index = [[i, j] for i, j in list(itertools.product(self.visible_row, self.visible_col))]
        #self.vision_index = [[i, j] for i, j in list(itertools.product([2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]))]
        # [0, 1, 2, 3], [上，下，左，右]

        self.obstacleH = [[self.rows // 2, col] for col in range(2, self.cols - 3)]
        self.obstacleV = [[row, self.center] for row in range(2, self.rows - 3)]

        # 两面墙
        self.wallH = [self.rows // 2, 2, self.cols - 3]
        self.wallV = [self.cols // 2, 2, self.rows - 3]
        self.obstacle_index = self.obstacleH + self.obstacleV

        self.action_space = [0, 1, 2, 3, 4]
        self.state = None
        self.obs = None
        self.array = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

        self.index = None
        self.penalty=penalty
        self.penalty_amount=penalty_amount
        self.path=path
        self.num=0

        self.stochastic = stochastic

        self.noisy_reward = noisy_reward
        self.noisy_reward_row = [i for i in range(0, self.rows // 2 - 3)]
        self.noisy_reward_index = [[i, j] for i, j in list(itertools.product(self.noisy_reward_row, self.visible_col))]

        self.vect = []
        for i in range(self.obs_range * 2):
            self.vect.append((-self.obs_range, -self.obs_range + i))
        for i in range(self.obs_range * 2):
            self.vect.append((-self.obs_range + i, self.obs_range))
        for i in range(self.obs_range * 2):
            self.vect.append((self.obs_range, self.obs_range - i))
        for i in range(self.obs_range * 2):
            self.vect.append((self.obs_range - i, -self.obs_range))


    def get_env_info(self):
        return {'state_shape': self.state_shape,
                'obs_shape': self.obs_shape,
                'subgoal_shape': self.subgoal_shape,
                'Goal_shape': self.Goal_shape,
                'episode_limit': self.episode_limit,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
                'n_subgoals': self.n_subgoals,
                'unit_dim': 0
                }


    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        np.save(self.path + '/heat_map_{}'.format(self.num), self.heat_map)
        # print(self.heat_map)
    def reset(self):
        self.num+=1
        self.index = [[0, 0], [self.rows - 1, self.cols - 1]]

        self.arrive1 = 0
        self.arrive2 = 0
        self.arrive3 = 0

        self._update_obs()
        self._episode_steps=0
        self.desired_goal = np.array([[5, 5], [5, 6]])

        return self.desired_goal

    def _update_obs(self):
        self.array[self.index[0][0]][self.index[0][1]] += 1
        self.array[self.index[1][0]][self.index[1][1]] += 1


        if self.map_name == 'wall4':
            pos_1 = self.index[0]
            pos_2 = self.index[1]
            self.obs_1 = [[0] * (self.obs_range * 2 + 1) for _ in range(self.obs_range * 2 + 1)]
            self.obs_2 = [[0] * (self.obs_range * 2 + 1) for _ in range(self.obs_range * 2 + 1)]
            
            # TODO obs墙壁赋值可优化
            for m in range(-self.obs_range, self.obs_range + 1):
                for n in range(-self.obs_range, self.obs_range + 1):
                    if pos_1[0]+m == self.wallH[0] and pos_1[1]+n <= self.wallH[2] and pos_1[1]+n >= self.wallH[1]:
                        self.obs_1[self.obs_range+m][self.obs_range+n] = -1
                    elif pos_1[1]+n == self.wallV[0] and pos_1[0]+m <= self.wallV[2] and pos_1[0]+m >= self.wallV[1]:
                        self.obs_1[self.obs_range+m][self.obs_range+n] = -1
                    if pos_2[0]+m == self.wallH[0] and pos_2[1]+n <= self.wallH[2] and pos_2[1]+n >= self.wallH[1]:
                        self.obs_2[self.obs_range+m][self.obs_range+n] = -1
                    elif pos_2[1]+n == self.wallV[0] and pos_2[0]+m <= self.wallV[2] and pos_2[0]+m >= self.wallV[1]:
                        self.obs_2[self.obs_range+m][self.obs_range+n] = -1
            if abs(pos_2[0] - pos_1[0]) <= self.obs_range and abs(pos_2[1] - pos_1[1]) <= self.obs_range:
                self.obs_1[self.obs_range + pos_2[0] - pos_1[0]][self.obs_range + pos_2[1] - pos_1[1]] = 1
                self.obs_2[self.obs_range + pos_1[0] - pos_2[0]][self.obs_range + pos_1[1] - pos_2[1]] = 1
        else:
            obs_1 = [[0 for _ in range(self.rows)], [0 for _ in range(self.cols)]]
            # obs_2 = obs_1.copy()
            import copy
            obs_2 = copy.deepcopy(obs_1)

            obs_1[0][self.index[0][0]] = 1
            obs_1[1][self.index[0][1]] = 1
            obs_1 = obs_1[0] + obs_1[1]

            obs_2[0][self.index[1][0]] = 1
            obs_2[1][self.index[1][1]] = 1
            obs_2 = obs_2[0] + obs_2[1]
            if self.map_name=="origin":
                if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                    temp = obs_1.copy()
                    obs_1 += obs_2.copy()
                    obs_2 += temp.copy()
                elif self.index[0] in self.vision_index:
                    obs_1 += obs_2.copy()
                    obs_2 += [0 for _ in range(self.rows + self.cols)]
                elif self.index[1] in self.vision_index:
                    obs_2 += obs_1.copy()
                    obs_1 += [0 for _ in range(self.rows + self.cols)]
                else:
                    obs_2 += [0 for _ in range(self.rows + self.cols)]
                    obs_1 += [0 for _ in range(self.rows + self.cols)]
            elif self.map_name=="full_observation":
                temp = obs_1.copy()
                obs_1 += obs_2.copy()
                obs_2 += temp.copy()
            elif self.map_name=="pomdp":
                if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                    temp = obs_1.copy()
                    obs_1 += obs_2.copy()
                    obs_2 += temp.copy()
                else:
                    obs_2 += [0 for _ in range(self.rows + self.cols)]
                    obs_1 += [0 for _ in range(self.rows + self.cols)]
            elif self.map_name == "reversed":
                # the second branch and the third branch are reversed.
                if self.index[0] in self.vision_index and self.index[1] in self.vision_index:
                    temp = obs_1.copy()
                    obs_1 += obs_2.copy()
                    obs_2 += temp.copy()
                elif self.index[0] in self.vision_index:
                    obs_2 += obs_1.copy()
                    obs_1 += [0 for _ in range(self.rows + self.cols)]
                elif self.index[1] in self.vision_index:
                    obs_1 += obs_2.copy()
                    obs_2 += [0 for _ in range(self.rows + self.cols)]
                else:
                    obs_2 += [0 for _ in range(self.rows + self.cols)]
                    obs_1 += [0 for _ in range(self.rows + self.cols)]


        #### add noise to state
        if self.noise:
            self.obs_1 += [np.random.normal() for i in range(self.noise_num)]
            self.obs_2 += [np.random.normal() for i in range(self.noise_num)]


        # self.state = pos_1 + self.obs_1 + pos_2 + self.obs_2
        self.obs = [np.hstack([np.array(pos_1), np.array(self.obs_1).flatten()]), np.hstack([np.array(pos_2), np.array(self.obs_2).flatten()])]

        self.state = [np.hstack([np.array(pos_1), np.array(self.obs_1).flatten(), np.array(pos_2), np.array(self.obs_2).flatten()])]
        # print(self.index)

    def get_state(self):
        return np.array(self.state)

    def get_obs(self):
        return self.obs


    def get_avail_actions(self):
        
        avail_actions=np.ones((2,5))
        for i in range(self.n_agents):
            current_obs = self.index[i]
            if current_obs[0] == 0:
                avail_actions[i,0] = 0
            if current_obs[0] == self.rows - 1:
                avail_actions[i,1] = 0
            if current_obs[1] == 0:
                avail_actions[i,2] = 0
            if current_obs[1] == self.cols - 1:
                avail_actions[i,3] = 0
            if current_obs[0] == self.wallH[0] - 1 and current_obs[1] >= self.wallH[1] and current_obs[1] <= self.wallH[2]:
                avail_actions[i,1] = 0
            elif current_obs[0] == self.wallH[0] + 1 and current_obs[1] >= self.wallH[1] and current_obs[1] <= self.wallH[2]:
                avail_actions[i,0] = 0
            elif current_obs[0] == self.wallH[0] and current_obs[1] == self.wallH[1]-1:
                avail_actions[i,3] = 0
            elif current_obs[0] == self.wallH[0] and current_obs[1] == self.wallH[2]+1:
                avail_actions[i,2] = 0

            if current_obs[1] == self.wallV[0] - 1 and current_obs[0] >= self.wallV[1] and current_obs[0] <= self.wallV[2]:
                avail_actions[i,3] = 0
            elif current_obs[1] == self.wallV[0] + 1 and current_obs[0] >= self.wallV[1] and current_obs[0] <= self.wallV[2]:
                avail_actions[i, 2] = 0
            elif current_obs[1] == self.wallV[0] and current_obs[0] == self.wallV[1]-1:
                avail_actions[i, 1] = 0
            elif current_obs[1] == self.wallV[0] and current_obs[0] == self.wallV[2]+1:
                avail_actions[i, 0] = 0
        # current_obs = self.index[1]
        # if current_obs[0] == 0:
        #     avail_actions[1,0] = 0
        # if current_obs[0] == self.rows - 1:
        #     avail_actions[1,1] = 0
        # if current_obs[1] == self.center:
        #     avail_actions[1,2] = 0
        # if current_obs[1] == self.cols - 1:
        #     avail_actions[1,3] = 0
        return avail_actions.tolist()
    
    def get_avail_subgoals(self):
        # TODO clid碰撞
        # 墙在中间时subgoal的确定
        avail_subgoals=np.ones((2, (self.obs_range + 2) * 4))
        for j, v in enumerate(self.vect):
            if self.obs_1[v[0]+self.obs_range][v[1]+self.obs_range] == -1:
                avail_subgoals[0, j] = 0
            if self.obs_2[v[0]+self.obs_range][v[1]+self.obs_range] == -1:
                avail_subgoals[1, j] = 0
        return avail_subgoals.tolist()

    def step(self, actions):
        # print('State is {}, action is {}'.format(self.state, actions))
        avail_actions = self.get_avail_actions()
        for idx in range(self.n_agents):
            action = actions[idx]
            if np.random.rand() < self.stochastic:
                sum = np.sum(avail_actions[idx])
                sampled_action = np.random.randint(sum)
                for i in range(4):
                    if avail_actions[idx][i] == 1:
                        if sampled_action == 0:
                            action = i
                            break
                        else:
                            sampled_action -= 1

            if action == 0:
                self.index[idx][0] -= 1 # row - 1
            elif action == 1:
                self.index[idx][0] += 1 # row + 1
            elif action == 2:
                self.index[idx][1] -= 1 # col - 1
            elif action == 3:
                self.index[idx][1] += 1 # col + 1

        # for i in range(self.rows):
        #     print(self.array[i])
        # print('*' * 100)
        self._update_obs()
        self._episode_steps +=1

        self.heat_map[self.index[0][0]][self.index[0][1]] += 1
        self.heat_map[self.index[1][0]][self.index[1][1]] += 1



        # print('Next state is {}'.format(self.state))
        if self.penalty:
            """ if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] != [self.rows // 2, self.center] and self.arrive1 == 0:
                self.arrive1 = 1
                reward = 30
                Terminated = False
                env_info = {'battle_won': False}
            elif self.index[0] != [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center] and self.arrive2 == 0:
                self.arrive2 = 1
                reward = 20
                Terminated = False
                env_info = {'battle_won': False} """
            """ if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] != [self.rows // 2, self.center]:
                reward = -10
                Terminated = False
                env_info = {'battle_won': False}
            elif self.index[0] != [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                reward = -10
                Terminated = False
                env_info = {'battle_won': False} """
            if self.index[1] == [self.rows // 2, self.center - 1] and self.index[0] == [self.rows // 2 + 1, self.center]:
                reward = 50
                Terminated = True
                env_info={'battle_won': True}
                """ if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                    # self.arrive1, self.arrive2 = 1, 1
                    reward = 50
                    Terminated=True
                    env_info={'battle_won': True}
                elif (self.index[0] == [3, 1] or self.index[1] == [3, 1]) and self.arrive1 == 0:
                    self.arrive1 = 1
                    reward = 5
                    Terminated = False
                    env_info = {'battle_won': False}
                elif (self.index[0] == [8, 10] or self.index[1] == [8, 10]) and self.arrive2 == 0:
                    self.arrive2 = 1
                    reward = 5
                    Terminated = False
                    env_info = {'battle_won': False}
                elif (self.index[0] == [10, 0] or self.index[1] == [10, 0]) and self.arrive3 == 0:
                    self.arrive3 = 1
                    reward = 10
                    Terminated = False
                    env_info = {'battle_won': False} """
            else:
                # reward = -self.penalty_amount
                reward = -self.penalty_amount
                Terminated = False
                env_info ={'battle_won': False}
        else:
            if self.index[0] == [self.rows // 2, self.center - 1] and self.index[1] == [self.rows // 2, self.center]:
                reward= 10
                Terminated=True
                env_info={'battle_won': True}
            else:
                reward = 0
                Terminated = False
                env_info ={'battle_won': False}

        if self.index[0] in self.noisy_reward_index or self.index[1] in self.noisy_reward_index:
            reward += np.random.randn() * self.noisy_reward

        if self._episode_steps >= self.episode_limit:
            Terminated= True
        if Terminated and self.path is not None:
            if self.num>1 and self.num % 500 == 0:
                self.save()
                print("save heat map")
                self.heat_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        return reward,Terminated,env_info

    def close(self):
        """Close StarCraft II."""
        pass



