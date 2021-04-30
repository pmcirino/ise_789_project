import numpy as np
from tqdm import tqdm
from output_files import save_outputs


class TemporalDifference():
    def __init__(self, inv_instance, learning_rate, alpha, epsilon, sample_size,output_name, w2=None):
        self.inv_instance = inv_instance
        self.eta = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.value_space = inv_instance.values
        self.action_dimension = inv_instance.actions_dimensions
        self.state_dimensions = inv_instance.values_dimensions
        self.x1_min = int(np.nanmin(self.value_space[1:,0]))
        self.x1_max = int(np.nanmax(self.value_space[1:, 0]))
        self.x2_min = int(np.nanmin(self.value_space[0,1:]))
        self.x2_max = int(np.nanmax(self.value_space[0, 1:]))
        self.q_table = np.zeros(tuple(self.action_dimension+self.state_dimensions))
        self.file_name = output_name
        self.w2 = w2

    def q_learning(self):
        epsilon = self.epsilon
        EVERY = 50000
        profits = []
        x1 = np.random.randint(self.x1_min, self.x1_max + 1)
        x2 = np.random.randint(self.x2_min, self.x2_max + 1)
        state = [x1,x2]
        for iteration in tqdm(range(self.sample_size)):
            if iteration % 100 == 0:
                epsilon = max(0.01, epsilon*self.epsilon)
                x1 = np.random.randint(self.x1_min, self.x1_max + 1)
                x2 = np.random.randint(self.x2_min, self.x2_max + 1)
                state = [x1, x2]
            if iteration % EVERY == 0 and iteration != 0:
                print('Average Profit', np.nanmean(profits[-EVERY:]))
                print('Epsilon', epsilon)

            action = self.inv_instance.get_td_action(state,self.q_table, epsilon)
            if self.w2 is not None:
                action[1]=max(self.w2-state[1],0)
            update = self.inv_instance.episode(state,action)
            reward = update['reward']
            index = tuple(action + [state[0]+self.x1_max,state[1]+self.x2_max])
            new_state = update['state']
            q_max = np.nanmax(self.q_table[:, :, new_state[0], new_state[1]])
            self.q_table[index] = self.q_table[index] + self.eta*(reward+self.alpha*q_max-self.q_table[index])
            state = new_state
            profits.append(reward)
        results = self.get_optimal_policy(q_table=self.q_table)

        return {'policy':results['policy'], 'values': results['values']}

    def get_optimal_policy(self, q_table):
        policy = np.zeros([2]+self.action_dimension)
        for x1 in range(-self.x1_max,self.x1_max+1):
            for x2 in range(-self.x2_max,self.x2_max+1):
                action = np.zeros(2)
                max_val = np.nanmax(q_table[:,:,x1+self.x1_max, x2+self.x2_max])
                opt_action = np.where(q_table[:,:,x1+self.x1_max, x2+self.x2_max]==max_val)
                if len(opt_action[0]) > 0:
                    action[0] = opt_action[0][0]
                    action[1] = opt_action[1][0]

                val_index = tuple([x1+self.x1_max+1,x2+self.x2_max+1])
                act_index = [x1+self.x1_max,x2+self.x2_max]
                self.value_space[val_index] = max_val
                policy[tuple([0]+act_index)] = action[0]
                policy[tuple([1] + act_index)] = action[1]
        result = {'policy':policy,'values': self.value_space}
        save_outputs.save_results(self.file_name, result)
        return {'policy':policy,'values': self.value_space}



