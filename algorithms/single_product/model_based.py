import numpy as np
from tqdm import tqdm
from output_files.save_outputs import save_results

class PolicyIteration():
    def __init__(self, inv_instance, epsilon, alpha, name):
        self.inv_instance = inv_instance
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_demand = int(self.inv_instance.max_demand)
        self.initial_policy = np.zeros(self.inv_instance.value_dimensions)
        self.name = name
        for x in range(-self.max_demand,self.max_demand+1):
            if x < 0:
                self.initial_policy[x+self.max_demand] = self.max_demand-x
    def policy_evaluation(self, policy):
        V_pi = [self.inv_instance.value_space]
        iteration = 0
        condition = False
        while not condition:
            v = np.zeros(self.inv_instance.value_dimensions)
            v[:, 0] = np.arange(-self.max_demand, self.max_demand + 1)
            for x in range(-self.max_demand, self.max_demand+1):
                val_index = int(x +self.max_demand)
                action_index = val_index
                action = int(policy[action_index,1])
                total_cost =self.inv_instance.get_expected_profit(x, action) +\
                            self.alpha * self.inv_instance.get_expected_future_profit(x, action,V_pi[iteration])
                v[val_index,1] = total_cost
            V_pi.append(v)
            iteration+=1
            norm = np.linalg.norm(np.subtract(V_pi[iteration],V_pi[iteration-1]))
            condition = norm < self.epsilon
        return V_pi[iteration]

    def policy_optimization(self):
        pi_n =[self.initial_policy]
        v_n =[self.policy_evaluation(self.initial_policy)]
        iteration = 0
        condition = False
        while not condition:
            new_pi = np.zeros(self.inv_instance.value_dimensions)
            for x in range(-self.max_demand,self.max_demand+1):
                candidates = []
                state_index = x + self.max_demand
                for y in range(max(0, -x), self.max_demand - x+ 1):
                    total_cost = self.inv_instance.get_expected_profit(x,y) +\
                        self.alpha*self.inv_instance.get_expected_future_profit(x,y, v_n[iteration])
                    candidates.append(total_cost)
                candidates = np.array(candidates)
                new_pi[state_index,1] = np.nanargmax(candidates)+max(-x,0)
                new_pi[state_index,0] = x
            pi_n.append(new_pi)
            print(new_pi, iteration)
            v_n.append(self.policy_evaluation(new_pi))
            iteration+=1
            condition = np.array_equal(pi_n[iteration],pi_n[iteration-1])
        result = {'values': v_n[-1],'policy':pi_n[-1]}
        save_results(self.name,result)
        return result


