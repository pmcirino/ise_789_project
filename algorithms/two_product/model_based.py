import numpy as np
from tqdm import tqdm
from environments.finite_demand import two_item
DEMAND_VALUES = np.array([np.arange(8), np.array([0,1,2,3,4])])
DEMAND_PROB = np.array([np.array([1/8]*8),np.array([0.2]*5)])
print(DEMAND_VALUES.shape)
h = np.array([1,1])
b = np.array([1,1])
p = np.array([15,15])
c = np.array([1,1])
theta = np.array([8,1])
gammas = np.array([.1, .8])

class PolicyIteration():
    def __init__(self, multi_product_instance,epsilon, alpha):
        self.inv_instance = multi_product_instance
        self.action_space = multi_product_instance.actions
        self.value_space = multi_product_instance.values
        self.epsilon = epsilon
        self.alpha = alpha
        self.x1_min = int(np.nanmin(self.value_space[1:,0]))
        self.x1_max = int(np.nanmax(self.value_space[1:, 0]))
        self.x2_min = int(np.nanmin(self.value_space[0,1:]))
        self.x2_max = int(np.nanmax(self.value_space[0, 1:]))
        self.expected_cost = multi_product_instance.calculate_expected_profits()
        self.initial_policy = np.zeros([2]+multi_product_instance.actions_dimensions)
        for x1 in range(self.x1_min,self.x1_max+1):
            for x2 in range(self.x2_min,self.x2_max+1):
                if x1 < 0:
                    self.initial_policy[0][x1+self.x1_max,x2+self.x2_max] = self.x1_max-x1
                if x2 < 0:
                    self.initial_policy[1][x1+self.x1_max,x2+self.x2_max] = self.x2_max - x2

    def policy_evaluation(self, policy):
        V_pi = [self.value_space]
        iteration = 0
        condition = False

        while not condition:
            v = np.zeros(self.inv_instance.values_dimensions)
            v[1:, 0] = np.arange(-self.x1_max,self.x1_max+1)
            v[0, 1:] = np.arange(-self.x2_max,self.x2_max+1)
            for x1 in range(self.x1_min,self.x1_max+1):
                for x2 in range(self.x2_min, self.x2_max + 1):
                    total_cost = 0
                    val_index = [x1+self.x1_max+1,x2+self.x2_max+1]
                    action_index = [x1+self.x1_max,x2+self.x2_max]
                    action = [int(policy[tuple([0]+action_index)]), int(policy[tuple([1]+action_index)])]
                    total_cost += self.expected_cost[tuple(action+val_index)]+\
                                  self.alpha * self.inv_instance.calculate_expected_future_profits(V_pi[iteration],[x1,x2],action)
                    v[tuple(val_index)] = np.around(total_cost,3)
            V_pi.append(v)
            iteration += 1
            norm = np.linalg.norm(np.subtract(V_pi[iteration],V_pi[iteration-1]))
            condition = norm < self.epsilon
        return V_pi[iteration]

    def policy_optimization(self):
        pi_n =[self.initial_policy]
        v_n =[self.policy_evaluation(self.initial_policy)]
        iteration = 0
        condition = False
        while not condition:
            new_pi = np.zeros([2]+self.inv_instance.actions_dimensions)
            for x1 in tqdm(range(self.x1_min,self.x1_max+1)):
                for x2 in range(self.x2_min,self.x2_max+1):
                    candidates=np.zeros([self.x1_max-x1+1,self.x2_max-x2+1])
                    state_index = [x1+self.x1_max+1, x2+self.x2_max+1]
                    for y1 in range(max(0,-x1), self.x1_max-x1+1):
                        for y2 in range(max(0, -x2), self.x2_max - x2+1):
                            action = [y1,y2]
                            total_cost = self.expected_cost[tuple(action+state_index)] +\
                                self.alpha * self.inv_instance.calculate_expected_future_profits(v_n[iteration],[x1,x2],action)
                            candidates[y1,y2] = total_cost
                    arr = candidates[max(0,-x1):,max(0,-x2):]
                    mins = np.unravel_index(np.nanargmax(arr), arr.shape)
                    opt = [mins[0] +max(0, -x1),mins[1] +max(0, -x2)]
                    index_0 = [0, x1+self.x1_max, x2+self.x2_max]
                    index_1 = [1,x1+self.x1_max,x2+self.x2_max]
                    new_pi[tuple(index_0)] = opt[0]
                    new_pi[tuple(index_1)] = opt[1]
            pi_n.append(new_pi)
            print(new_pi)
            iteration += 1
            v_n.append(self.policy_evaluation(pi_n[iteration]))
            condition = np.array_equal(pi_n[iteration][0], pi_n[iteration - 1][0]) and \
                        np.array_equal(pi_n[iteration][1], pi_n[iteration - 1][1])
            print(iteration)
        return {'values':v_n,'policy':pi_n}

    def reaction_policy_iteration(self, w2):
        pi_n = [self.initial_policy]
        v_n = [self.policy_evaluation(self.initial_policy)]
        iteration = 0
        condition = False
        while not condition:
            new_pi = np.zeros([2] + self.inv_instance.actions_dimensions)
            for x1 in tqdm(range(self.x1_min, self.x1_max + 1)):
                for x2 in range(self.x2_min, self.x2_max + 1):
                    candidates = np.zeros([self.x1_max - x1 + 1, self.x2_max - x2 + 1])
                    state_index = [x1 + self.x1_max + 1, x2 + self.x2_max + 1]
                    for y1 in range(max(0, -x1), self.x1_max - x1 + 1):
                        y2 = 0
                        if x2<w2:
                            y2 = w2 - x2
                        action = [y1, y2]
                        total_cost = self.expected_cost[tuple(action + state_index)] + \
                                     self.alpha * self.inv_instance.calculate_expected_future_profits(
                            v_n[iteration], [x1, x2], action)
                        candidates[y1, y2] = total_cost
                    arr = candidates[max(0, -x1):, max(0, -x2):]
                    mins = np.unravel_index(np.nanargmax(arr), arr.shape)
                    opt = [mins[0] + max(0, -x1), mins[1] + max(0, -x2)]
                    index_0 = [0, x1 + self.x1_max, x2 + self.x2_max]
                    index_1 = [1, x1 + self.x1_max, x2 + self.x2_max]
                    new_pi[tuple(index_0)] = opt[0]
                    new_pi[tuple(index_1)] = opt[1]
            pi_n.append(new_pi)
            print(new_pi)
            iteration += 1
            v_n.append(self.policy_evaluation(pi_n[iteration]))
            condition = np.array_equal(pi_n[iteration][0], pi_n[iteration - 1][0]) and \
                        np.array_equal(pi_n[iteration][1], pi_n[iteration - 1][1])
            print(iteration)
        return {'values': v_n, 'policy': pi_n}





test = two_item.TwoItem(DEMAND_VALUES,DEMAND_PROB, h, b, p, c, theta, gammas)

pi = PolicyIteration(test, 0.01, 0.9)
test2 = pi.reaction_policy_iteration(1)
np.printoptions(supress=True)
print(test2['values'][-1])
print(test2['policy'][-1])