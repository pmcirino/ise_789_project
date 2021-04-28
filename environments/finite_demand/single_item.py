import numpy as np
import tqdm
DEMAND_VALUES = np.arange(0, 5)
DEMAND_PROB = np.array([0.2]*5)
class SingleItem():
    def __init__(self,demand_values, demand_probabilities, h, b, p, c, theta):
        self.demand_values = demand_values
        self.demand_probabilities = demand_probabilities
        self.h = h
        self.b = b
        self.p = p
        self.c = c
        self.theta = theta
        self.max_demand = np.nanmax(demand_values)
        self.action_space = np.zeros([2*self.max_demand+1,2])
        self.action_space[:,0] = np.arange(-self.max_demand, self.max_demand+1)
        self.value_space = np.zeros([2*self.max_demand+1,2])
        self.value_space[:, 0] = np.arange(-self.max_demand, self.max_demand + 1)

    ### Model Based Methods ###
    def get_probability(self,d):
        index = np.where(self.demand_values == d)
        return self.demand_probabilities[index]

    def get_expected_profit(self, x, y):
        I = x + y
        # calculate expiration cost
        cost = self.c*y
        revenue = 0
        for d in range(0,x):
            cost += self.get_probability(d)*(x-d)*self.theta
            revenue = self.get_probability(d)*self.p * d
        # calculate expected holding cost
        for d in range(x,I):
            cost += self.get_probability(d)*(I-d)*self.h
            revenue = self.get_probability(d) * self.p * d
        # calculate expected backorder cost
        for d in range(I,self.max_demand+1):
            cost += self.get_probability(d) * (d-I) * self.b
            revenue = self.get_probability(d) * self.p * I

        return revenue-cost

    def get_expected_future_profit(self, x, y, value_table):
        profit = 0
        for d in range(0, x):
            index = np.where(value_table[:, 0]== int(y))
            profit += value_table[index,2]*self.demand_probabilities(d)
        for d in range(x,self.max_demand+1):
            index = np.where(value_table[:,0]==int(x+y-d))
            profit += value_table[index,2]*self.demand_probabilities(d)
        return profit

    ### Reinforcement Learning Based Methods ###
    def get_action(self, policy):
        return 'get action'

    def get_state(self):
        return 'get state'

    def  calculate_reward(self):
        return 'calculate reward'




test = SingleItem(DEMAND_VALUES,DEMAND_PROB, 1,1,1,1,1)

print(test.get_probability(2))
