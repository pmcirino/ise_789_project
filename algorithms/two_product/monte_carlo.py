import numpy as np
from tqdm import tqdm
from environments.finite_demand import two_item
from environments.infinte_demand.two_item import TwoItem as PoissonTwoItem
from output_files.save_outputs import save_results

class PolicyIteration():
    def __init__(self, inv_instance, num_episode,episode_length,epsilon, alpha,name,w2=None):
        self.inv_instance = inv_instance
        self.num_episodes = num_episode
        self.episode_length = episode_length
        self.epsilon = epsilon
        self.alpha = alpha
        self.value_space = inv_instance.values
        self.action_dimension = inv_instance.actions_dimensions
        self.state_dimensions = inv_instance.values_dimensions
        self.x1_min = int(np.nanmin(self.value_space[1:,0]))
        self.x1_max = int(np.nanmax(self.value_space[1:, 0]))
        self.x2_min = int(np.nanmin(self.value_space[0,1:]))
        self.x2_max = int(np.nanmax(self.value_space[0, 1:]))
        self.u_dimensions = tuple(self.action_dimension+self.state_dimensions)
        self.initial_policy = np.zeros([2]+inv_instance.actions_dimensions)
        self.w2 = w2
        self.file_name = name
        for x1 in range(self.x1_min,self.x1_max+1):
            for x2 in range(self.x2_min,self.x2_max+1):
                if x1 < 0:
                    self.initial_policy[0][x1+self.x1_max,x2+self.x2_max] = self.x1_max-x1
                if w2 is not None:
                    self.initial_policy[1][x1 + self.x1_max, x2 + self.x2_max] = max(w2 - x2,0)
                elif x2 < 0:
                    self.initial_policy[1][x1+self.x1_max,x2+self.x2_max] = self.x2_max - x2
    def policy_evaluation(self,policy):
        print(self.u_dimensions)
        print(self.initial_policy.shape)
        u = np.zeros(self.u_dimensions)
        u_count = np.zeros(self.u_dimensions)
        v_pi = np.zeros(self.u_dimensions)
        epsilon = self.epsilon
        for episode in tqdm(range(self.num_episodes)):
            states =[]
            rewards =[]
            actions =[]
            x1 = np.random.randint(self.x1_min,self.x1_max+1)
            x2 = np.random.randint(self.x2_min,self.x2_max+1)
            state = [x1,x2]
            # run an episode
            G = 0
            Gs =[]
            epsilon=min(0.01,epsilon*self.epsilon)
            for t in range(self.episode_length):
                action = self.inv_instance.get_action(policy,state,epsilon)
                if self.w2 is not None:
                    action[1]=max(self.w2-state[1],0)
                if action[0] < -state[0] or action[1]< -state[1]:
                    print('error')
                result = self.inv_instance.episode(state,action)
                reward = result['reward']
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = result['state']
            # update table
            T = len(rewards)
            for t in range(T-1,-1,-1):
                G = self.alpha*G + rewards[t]
                state_index =[states[t][0]+self.x1_max, states[t][1]+self.x2_max]
                index = tuple(actions[t]+state_index)
                Gs.append(G)
                u[index] = u[index]+G
                u_count[index]=u_count[index]+1
        v_pi = np.divide(u,u_count)
        return v_pi

    def policy_update(self,v_pi):
        new_policy = np.zeros([2]+self.action_dimension)
        count = 0
        values = np.zeros(self.state_dimensions)
        values[1:,0] = np.arange(-self.x1_max,self.x1_max+1)
        values[0,1:] = np.arange(-self.x2_max, self.x2_max + 1)
        for x1 in range(self.x1_min,self.x1_max+1):
            for x2 in range(self.x2_min,self.x2_max+1):
                index1 = int(x1+self.x1_max)
                index2 = int(x2+self.x2_max)
                max_profit = np.nanmax(v_pi[:,:,index1,index2])
                new_action = np.where(v_pi[:,:,index1,index2]==max_profit)
                action =np.array([0,0])
                values[index1+1,index2+1] = max_profit
                if len(new_action[0]) ==0 :
                    action[0]=self.x1_max-x1
                    action[1]=self.x2_max-x2
                    count += 1
                else:
                    action[0] = new_action[0][0]
                    action[1] = new_action[1][0]
                if action[0]+x1 < 0:
                    print('error')
                if action[1] + x2 <0:
                    print('error')
                new_policy[0][index1,index2] = action[0]
                new_policy[1][index1, index2] = action[1]

        return {'policy': new_policy, 'values': values}
        # Reaction Functions

    def reaction_policy_update(self, v_pi):
        new_policy = np.zeros([2] + self.action_dimension)
        count = 0
        values = np.zeros(self.state_dimensions)
        values[1:, 0] = np.arange(-self.x1_max, self.x1_max + 1)
        values[0, 1:] = np.arange(-self.x2_max, self.x2_max + 1)
        for x1 in range(self.x1_min, self.x1_max + 1):
            for x2 in range(self.x2_min, self.x2_max + 1):
                index1 = int(x1 + self.x1_max)
                index2 = int(x2 + self.x2_max)
                max_profit = np.nanmax(v_pi[:, :, index1, index2])
                new_action = np.where(v_pi[:, :, index1, index2] == max_profit)
                action = np.array([0, 0])
                values[index1 + 1, index2 + 1] = max_profit
                if len(new_action[0]) == 0:
                    action[0] = self.x1_max - x1
                    action[1] = max(self.w2 - x2,0)
                    count += 1
                else:
                    action[0] = new_action[0][0]
                    action[1] = max(self.w2 - x2,0)
                if action[0] + x1 < 0:
                    print('error')
                if action[1] + x2 < 0:
                    print('error')
                new_policy[0][index1, index2] = action[0]
                new_policy[1][index1, index2] = action[1]
        return {'policy': new_policy, 'values': values}

    def policy_iteration(self):
        pi_n = [self.initial_policy]
        v_n = [self.policy_evaluation(self.initial_policy)]
        values = []
        condition = False
        iteration = 0
        while not condition and iteration<5:
            if self.w2 is not None:
                update=self.reaction_policy_update(v_n[iteration])
            else:
                update = self.policy_update(v_n[iteration])
            pi_n.append(update['policy'])
            v_n.append(self.policy_evaluation(update['policy']))
            values.append(update['values'])
            condition = np.array_equal(pi_n[iteration][0], pi_n[iteration - 1][0]) and \
                        np.array_equal(pi_n[iteration][1], pi_n[iteration - 1][1])
            iteration +=1
            print(iteration)
            print(pi_n[iteration])
        result = {'policy': pi_n[-1],'values': values[-1]}
        save_results(self.file_name,result)
        return {'policy': pi_n,'values': values}





