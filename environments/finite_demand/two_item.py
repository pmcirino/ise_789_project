import numpy as np
import tqdm
#DEMAND_VALUES = np.array([np.arange(0, 5), np.arange(0, 5)])
##DEMAND_PROB = np.array([np.array([0.2]*5),np.array([0.2]*5)])
#h = np.array([.1,.1])
#b = np.array([1,1])
#p = np.array([8,8])
#c = np.array([1.3,1])
#theta = np.array([.1,.1])
#gammas = np.array([.7, .8])

class TwoItem():
    def __init__(self,demand_values, demand_probabilities, h, b, p, c, theta,substitution_rates):
        self.demand_values = demand_values
        self.demand_probabilities = demand_probabilities
        self.h = h
        self.b = b
        self.p = p
        self.c = c
        self.theta = theta
        self.gamma1 = substitution_rates[0]
        self.gamma2 = substitution_rates[1]
        self.max_demands = np.zeros(2)
        self.actions_dimensions = []
        self.values_dimensions = []
        self.inv_range = []

        for i in range(2):
            self.max_demands[i] = np.nanmax(demand_values[i])
            self.inv_range.append(np.arange(-self.max_demands[i],self.max_demands[i]+1))
            self.actions_dimensions.append(int(2*self.max_demands[i]+1))
            self.values_dimensions.append(int(2*self.max_demands[i]+2))
        self.actions = np.zeros(self.actions_dimensions+[2])
        self.values = np.zeros(self.values_dimensions)
        self.q_table = np.zeros(self.actions_dimensions+self.values_dimensions)
        self.expected_profit = np.zeros(self.actions_dimensions+self.values_dimensions)
        self.values[1:,0] = self.inv_range[0]
        self.values[0,1:] =self.inv_range[1]
        for i in range(self.actions_dimensions[0]):
            for j in range(self.actions_dimensions[1]):
                self.q_table[i,j,1:, 0] = self.inv_range[0]
                self.q_table[i,j,0, 1:] = self.inv_range[1]
                self.expected_profit[i,j, 1:, 0] = self.inv_range[0]
                self.expected_profit[i, j, 0, 1:] = self.inv_range[1]

    def get_probability(self,d, item_no):
        if d in self.demand_values[item_no-1]:
            index = np.where(self.demand_values[item_no-1] == d)
            prob = self.demand_probabilities[item_no-1][index]
        else:
            prob = 0
        return prob

    def get_expected_expiration_cost(self,state,action):
        x1 = int(state[0])
        x2 = int(state[1])
        y1 = int(action[0])
        y2 = int(action[1])
        ### Case where both goods face expiration
        cost = 0
        total_prob = 0
        for d1 in range(x1):
            for d2 in range(x2):
                probability = self.get_probability(d1,1) * self.get_probability(d2,2)
                cost += (self.theta[0]*(x1-d1)+self.theta[1]*(x2-d2))*probability
                total_prob+=probability

        ### Case where item 1 faces expiration and item 2 demand is met
        for d1 in range(x1):
            for d2 in range(x2,x2+y2+1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[0] * (x1 - d1) * probability
                total_prob += probability
        ### Case where item 1 is used to substitute for item 2
        for d1 in range(x1):
            for d2 in range(x2+y2+1,x2+y2+1 + int((x1-d1)/self.gamma1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[0] * (x1 - d1-self.gamma1*(d2-x2+y2)) * probability
                total_prob += probability

        ### Case where item 2 faces expiration and item 1 demand is met
        for d2 in range(x2):
            for d1 in range(x1,x1+y1+1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[1] * (x2 - d2) * probability

        ### Case where item 2 is used to substitute for item 1
        for d2 in range(x2):
            for d1 in range(x1+y1+1,x1+y1+1+int((x2-d2)/self.gamma2) ):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[1] * (x2 - d2) * probability
                total_prob += probability
        return cost

    def get_expected_inventory_cost(self,state,action):
        x1 = int(state[0])
        x2 = int(state[1])
        y1 = int(action[0])
        y2 = int(action[1])
        # Case where both goods face expiration
        revenue = 0
        cost = 0
        total_prob = 0
        for d1 in range(x1):
            for d2 in range(x2):
                probability = self.get_probability(d1,1) * self.get_probability(d2,2)
                cost += (self.h[0]*y1 + self.h[1]*y2)*probability
                revenue += (d1*self.p[0] +d2*self.p[1])*probability
                total_prob += probability
        # Case where item 1 faces expiration
        for d1 in range(x1):
            for d2 in range(x2, x2+y2+1):
                probability = self.get_probability(d1,1) * self.get_probability(d2,2)
                cost += (self.h[0]*y1 + self.h[1]*(y2+x2-d2))*probability
                revenue += (d1 * self.p[0] + d2 * self.p[1]) * probability
                total_prob += probability
        # Case where item 1 faces expiration and we have substitution for 2
        for d1 in range(x1):
            for d2 in range(x2+y2+1,x2+y2+1 + int((x1-d1)/self.gamma1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.h[0] * y1 + self.b[1] *(1-self.gamma1) * (d2 - x2-y2)) * probability
                revenue += (d1 * self.p[0] + (x1+y1+self.gamma1*(d2 - x2 - y2)) * self.p[1]) * probability
                total_prob += probability

        # case where item 2 faces expiration
        for d2 in range(x2):
            for d1 in range(x1, x1+y1+1):
                probability = self.get_probability(d1,1) * self.get_probability(d2, 2)
                cost += (self.h[0]*(x1+y1-d1) + self.h[1]*y2)*probability
                revenue += (d1 * self.p[0] + d2 * self.p[1]) * probability
                total_prob += probability

        # Case where item 2 faces expiration and we have substitution for 1
        for d2 in range(x2):
            for d1 in range(x1+y1+1,x1+y1+1 + int((x2-d2/self.gamma2))):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.b[0] * (1-self.gamma2) * (d1 - x1-y1)+self.h[1] * y2) * probability
                revenue += (d2 * self.p[1] + (x2+y2+self.gamma2*(d1 - x1 - y1)) * self.p[0]) * probability
                total_prob += probability

        # Case where both have holding cost
        for d1 in range(x1, x1+y1+1):
            for d2 in range(x2, x2+y2+1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.h[0]*(x1+y1-d1)+self.h[1]*(x2+y2-d2))*probability
                revenue += (d1 * self.p[0] + d2 * self.p[1]) * probability
                total_prob += probability

        # Case where 1 has holding cost and is substituted for 2
        for d1 in range(x1,x1+y1+1):
            for d2 in range(x2+y2+1, x2+y2+1 + int((x1+y1-d1)/self.gamma1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.h[0]*(x1+y1-d1-self.gamma1*(d2-x2-y2))+self.b[1]*(1-self.gamma1)*(d2-x2-y2))*probability
                revenue += (d1 * self.p[0] + (x1 + y1 + self.gamma1 * (d2 - x2 - y2)) * self.p[1]) * probability
                total_prob += probability

        # case where 1 is subbed for 2 and all units are used
        for d1 in range(x1, x1 + y1 + 1):
            for d2 in range(x2 + y2 + 1 + int((x1 + y1 - d1) / self.gamma1), int(self.max_demands[1]+1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.b[1] * (d2+ d1 - x2 - y2-x1-y1) * probability
                revenue += (d1 * self.p[0] + (x1 + y1 + x2 + y2 - d1) * self.p[1]) * probability
                total_prob += probability

        # case where 2 is subbed for 1
        for d2 in range(x2, x2+y2+1):
            for d1 in range(x1+y1+1, x1+y1+1 + int((x2+y2-d2)/self.gamma2)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.b[0]*(1-self.gamma2)*(d1-x1-y1)+self.h[0]*(x2+y2-d2-self.gamma2*(d1-x1-y1))) * probability
                revenue += (d2 * self.p[1] + (x2 + y2 + self.gamma2 * (d1 - x1 - y1)) * self.p[0]) * probability
                total_prob += probability

        # Case where 2 is subbed for 1 and all units are sold
        for d2 in range(x2, x2+y2+1):
            for d1 in range(x1+y1+1 + int((x2+y2-d2)/self.gamma2),int(self.max_demands[0]+1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.b[0]*(d1+d2-x2-y2-x1-y1)*probability
                revenue += (d2 * self.p[1] + (x2 + y2 + x1+y1-d2) * self.p[0]) * probability
                total_prob += probability

        for d1 in range(x1+y1+1,int(self.max_demands[0]+1)):
            for d2 in range(x2 + y2 + 1, int(self.max_demands[1]+1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.b[0] * (d1 + x1 - y1) + self.b[1]*(d2 - x2 - y2) * probability
                revenue += ((x1+y1) * self.p[0] + (x2+y2) * self.p[1]) * probability
                total_prob += probability
        return revenue-cost

    def calculate_expected_profits(self):
        for x1 in self.inv_range[0]:
            for x2 in self.inv_range[1]:
                for y1 in range(int(self.max_demands[0]-x1+1)):
                    for y2 in range(int(self.max_demands[1] - x2 + 1)):
                        index = (y1, y2, int(x1+self.max_demands[0]+1), int(x2+self.max_demands[1]+1))
                        self.expected_profit[index] = -y1*self.c[0]-y2*self.c[0] - self.get_expected_expiration_cost([int(x1),int(x2)],[y1,y2]) +\
                            self.get_expected_inventory_cost([int(x1),int(x2)],[y1,y2])
        return self.expected_profit

    def calculate_expected_future_profits(self, value_table, state, action):
        x1 = int(state[0])
        x2 = int(state[1])
        y1 = int(action[0])
        y2 = int(action[1])
        # Case where both goods face expiration
        revenue = 0
        total_prob = 0
        for d1 in range(x1):
            for d2 in range(x2):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [y1+1,y2+1]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where item 1 faces expiration
        for d1 in range(x1):
            for d2 in range(x2, x2 + y2 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [y1+1, d2-x2-y2+1]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where item 1 faces expiration and we have substitution for 2
        for d1 in range(x1):
            for d2 in range(x2 + y2 + 1, x2 + y2 + 1 + int((x1 - d1) / self.gamma1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [1 + y1, 1+int((1 - self.gamma1) * (d2 - x2 - y2))]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # case where item 2 faces expiration
        for d2 in range(x2):
            for d1 in range(x1, x1 + y1 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [1+x1+y1 - d1, 1+ y2]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where item 2 faces expiration and we have substitution for 1
        for d2 in range(x2):
            for d1 in range(x1 + y1 + 1, x1 + y1 + 1 + int((x2 - d2 / self.gamma2))):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [1+ int((1 - self.gamma2) * (d1 - x1 - y1)),1 + y2]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where both have holding cost
        for d1 in range(x1, x1 + y1 + 1):
            for d2 in range(x2, x2 + y2 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [1+x1 + y1 - d1, 1+x2 + y2 - d2]
                total_prob += probability
                if probability >0:
                    revenue += value_table[tuple(index)] * probability

        # Case where 1 has holding cost and is substituted for 2
        for d1 in range(x1, x1 + y1 + 1):
            for d2 in range(x2 + y2 + 1, x2 + y2 + 1 + int((x1 + y1 - d1) / self.gamma1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [int(x1 + y1 - d1 - self.gamma1 * (d2 - x2 - y2)), int((1 - self.gamma1) * (
                            d2 - x2 - y2))]
                total_prob += probability
                if probability >0:
                    revenue += value_table[tuple(index)] * probability

        # case where 1 is subbed for 2 and all units are used
        for d1 in range(x1, x1 + y1 + 1):
            for d2 in range(x2 + y2 + 1 + int((x1 + y1 - d1) / self.gamma1), int(self.max_demands[1])):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [0, int(d2 + d1 - x2 - y2 - x1 - y1)]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # case where 2 is subbed for 1
        for d2 in range(x2, x2 + y2 + 1):
            for d1 in range(x1 + y1 + 1, x1 + y1 + 1 + int((x2 + y2 - d2) / self.gamma2)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [int((1 - self.gamma2) * (d1 - x1 - y1)),int(x2 + y2 + self.gamma2 * (d1 - x1 - y1))]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where 2 is subbed for 1 and all units are sold
        for d2 in range(x2, x2 + y2 + 1):
            for d1 in range(x1 + y1 + 1 + int((x2 + y2 - d2) / self.gamma2), int(self.max_demands[0])):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [d1 + d2 - x2 - y2 - x1 - y1,0]
                total_prob += probability
                if probability >0:
                    revenue += value_table[tuple(index)] * probability

        for d1 in range(x1 + y1 + 1, int(self.max_demands[0])):
            for d2 in range(x2 + y2 + 1, int(self.max_demands[1])):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index=[d1 + x1 - y1, d2 - x2 - y2]
                revenue += value_table[tuple(index)] * probability
                total_prob += probability
        return revenue



