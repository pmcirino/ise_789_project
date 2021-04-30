import numpy as np
from scipy.stats import poisson
import tqdm
DEMAND_VALUES = np.arange(0, 5)
DEMAND_PROB = np.array([0.2]*5)
class SingleItem():
    def __init__(self,demand_rate, max_demand, h, b, p, c, theta):
        self.mu = demand_rate
        self.max_demand = int(poisson.ppf(max_demand,self.mu))
        self.demand_values = np.arange(self.max_demand+1)
        self.demand_probabilities = poisson.pmf(self.demand_values,self.mu)
        print(self.demand_values,self.demand_probabilities)
        self.h = h
        self.b = b
        self.p = p
        self.c = c
        self.theta = theta
        self.actions_dimensions = [2*self.max_demand+1,2]
        self.value_dimensions = [2*self.max_demand+1,2]
        self.action_space = np.zeros([2*self.max_demand+1,2])
        self.action_space[:,0] = np.arange(-self.max_demand, self.max_demand+1)
        self.value_space = np.zeros([2*self.max_demand+1,2])
        self.value_space[:, 0] = np.arange(-self.max_demand, self.max_demand + 1)

    ### Model Based Methods ###
    def get_probability(self,d):
        prob = 0
        if d in self.demand_values:
            index = np.where(self.demand_values == d)
            prob = self.demand_probabilities[index]
        return prob

    def get_expected_profit(self, x, y):
        I = x + y
        # calculate expiration cost
        cost = self.c*y
        revenue = 0
        for d in range(0,x):
            probability = self.get_probability(d)
            cost += probability*(x-d)*self.theta + probability*(y)*self.h
            revenue += probability*self.p * d
        # calculate expected holding cost
        for d in range(x,I):
            probability = self.get_probability(d)
            cost += probability*(I-d)*self.h
            revenue += probability * self.p * d
        # calculate expected backorder cost
        for d in range(I,self.max_demand+1):
            probability = self.get_probability(d)
            cost += probability * (d-I) * self.b
            revenue += probability * self.p * I

        return revenue-cost

    def get_expected_future_profit(self, x, y, value_table):
        profit = 0
        for d in range(0, x):
            index = np.where(value_table[:, 0]== int(y))
            probability = self.get_probability(d)
            if probability != 0:
                profit += value_table[index,1]*probability
        for d in range(x,self.max_demand+1):
            index = np.where(value_table[:,0]==int(x+y-d))
            probability = self.get_probability(d)
            if probability != 0:
                profit += value_table[index,1]*probability
        return profit

    ### Reinforcement Learning Based Methods ###
    def get_action(self, policy, epsilon):
        return 'get action'

    def get_state(self, x, y):
        d = np.random.choice(self.demand_values, 1, list(self.demand_probabilities))
        if d <= x:
            state = y
        else:
            state = x+y-d
        return state

    def  calculate_reward(self):
        return 'calculate reward'



