import numpy as np
from scipy.stats import poisson
from tqdm import tqdm
#DEMAND_VALUES = np.array([np.arange(0, 5), np.arange(0, 5)])
##DEMAND_PROB = np.array([np.array([0.2]*5),np.array([0.2]*5)])
#h = np.array([.1,.1])
#b = np.array([1,1])
#p = np.array([8,8])
#c = np.array([1.3,1])
#theta = np.array([.1,.1])
#gammas = np.array([.7, .8])

class TwoItem():
    def __init__(self,demand_rates,max_demand, h, b, p, c, theta,substitution_rates):
        self.mu1 = demand_rates[0]
        self.mu2 = demand_rates[1]
        self.max_demands = np.zeros(2)
        self.max_demands[0] = int(poisson.ppf(max_demand,self.mu1))
        self.max_demands[1] = int(poisson.ppf(max_demand,self.mu2))
        self.demand_values = np.array([np.arange(-self.max_demands[0],self.max_demands[0]+1),
                                       np.arange(-self.max_demands[1],self.max_demands[1]+1)])
        print(self.demand_values)
        self.h = h
        self.b = b
        self.p = p
        self.c = c
        self.theta = theta
        self.gamma1 = substitution_rates[0]
        self.gamma2 = substitution_rates[1]

        self.actions_dimensions = []
        self.values_dimensions = []
        self.inv_range = []
        for i in range(2):
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
    # Temporal Difference Support methods
    def get_td_action(self, state, q_table, epsilon):
        k = np.random.random()
        action = np.zeros(2)
        #print(state)
        if k > epsilon and not np.all(q_table[:,:,int(state[0]+self.max_demands[0]),int(state[1]+self.max_demands[1])] <= 0):
            max_profit = np.nanmax(q_table[:,:,int(state[0]+self.max_demands[0]),int(state[1]+self.max_demands[1])])
            new_action = np.where(q_table[:,:,int(state[0]+self.max_demands[0]),int(state[1]+self.max_demands[1])]==max_profit)
            if len(new_action[0]) > 0:
                action[0] = new_action[0][0]
                action[1] = new_action[1][0]
            else:
                y1 = np.random.randint(max(0, -int(state[0])), int(self.max_demands[0] - state[0] + 1))
                y2 = np.random.randint(max(0, -int(state[1])), int(self.max_demands[1] - state[1] + 1))
                action = np.array([y1, y2])
        else:
            y1 = np.random.randint(max(0,-int(state[0])),int(self.max_demands[0]-state[0]+1))
            y2 = np.random.randint(max(0,-int(state[1])),int(self.max_demands[1]-state[1]+1))
            action = np.array([y1,y2])
        if isinstance(action[0],list):
            action = action[0]
        return list(action.astype(int))

    # Monte-Carlo Support Methods
    def get_action(self,policy,state,epsilon):
        k = np.random.random()
        action = np.zeros(2)
        #print(state)
        if k > epsilon:
            action[0] = policy[0,int(state[0]+self.max_demands[0]),int(state[1]+self.max_demands[1])]
            action[1] = policy[1,int(state[0]+self.max_demands[0]),int(state[1]+self.max_demands[1])]
        else:
            y1 = np.random.randint(max(0,-int(state[0])),int(self.max_demands[0]-state[0]+1))
            y2 = np.random.randint(max(0,-int(state[1])),int(self.max_demands[1]-state[1]+1))
            action = np.array([y1,y2])
        if isinstance(action[0],list):
            action = action[0]
        return list(action.astype(int))

    def episode(self, state, action):
        x1 = int(state[0])
        x2 = int(state[1])
        y1 = int(action[0])
        y2 = int(action[1])
        I1 = x1+y1
        I2 = x2+y2
        d1 = min(int(np.random.poisson(self.mu1,1)),self.max_demands[0])
        d2 = min(int(np.random.poisson(self.mu2,1)),self.max_demands[1])
        cost = 0
        revenue = 0
        count =0
        new_state =0
        max_substitution1 = int((I1-d1)/self.gamma1)
        max_substitution2 = int((I2-d2)/self.gamma2)
        if d1 <= I1 and d2 <= I2:
            cost += self.theta[0]*max(x1-d1,0) + self.theta[1]*max(x2,d2,0) # expiration cost
            cost += self.h[0] * min(y1,I1-d1) + self.h[1] * min(y2,I2-d1)  # holding cost
            revenue += d1*self.p[0] + d2*self.p[1]
            new_state = [int(min(y1, I1-d1)), int(min(y2, I2-d2))]
            count+=1
        if d1 <= I1 and I2<d2 <=I2+max_substitution1:
            cost += self.theta[0] * max(x1 - int(d1-self.gamma1*(d2-I2)),0) # expiration cost
            cost += self.h[0] * min(y1,I1-d1)  # holding cost
            cost += self.b[1]*int((1-self.gamma1)*(d2-I2)) # backorder cost
            revenue += d1 * self.p[0] + I2 * self.p[1] + int(d1-self.gamma1*(d2-I2))*self.p[1]
            new_state = [int(min(y2, int(I1-d1-self.gamma1*(d2-I2)))), -int((1-self.gamma1)*(d2-I2))]
            count += 1
        if d1 <= I1 and I2+max_substitution1 < d2 <= self.max_demands[1]:
            cost += self.b[1]* int(d2+d1-I2-I1) # backorder cost
            revenue += d1*self.p[0] + (I1+I2-d1)*self.p[1]
            new_state = [0, -int(d2+d1-I2-I1)]
            count += 1
        if I1<d1<=I1 + max_substitution2 and d2<=I2:
            cost += self.theta[1]*min(0,x2-d2) # expiration cost
            cost += self.h[1] * (I2-d2-int(self.gamma2*(d1-I1)))  # holding cost
            cost += self.b[0]*int((1-self.gamma2)*(d1-I1))
            revenue += I1*self.p[0] + d2*self.p[1]+ self.p[0]*int(self.gamma2*(d1-I1))
            new_state = [-int((1-self.gamma2)*(d1-I1)), I2-d2-int(self.gamma2*(d1-I1))]
            count += 1
        if I1 +max_substitution2 <d1<=self.max_demands[0] and d2<=I2:
            cost += self.b[0] * int(d1+d2-I1-I2)  # holding cost
            revenue += I1*self.p[0] + d2*self.p[1] + (I2+I1-d2)*self.p[0]
            new_state = [-int(d1+d2-I1-I2), 0]
            count += 1

        if d1 > I1 and d2 >I2:
            cost +=self.b[0]*(d1-I1) + self.b[1] * (d2 - I2)  # backorder cost
            revenue += self.p[0]*I1 + self.p[1]*I2
            new_state = [-int(d1-I1), -int(d2 - I2)]
            count += 1
        if new_state[0]>self.max_demands[0]:
            new_state[0] = int(self.max_demands[0])
        if new_state[1] > self.max_demands[1]:
            new_state[1] = int(self.max_demands[1])
        if not isinstance(new_state,list):
            new_state = list(new_state)
        profit = revenue - cost -self.c[0]*y1 - self.c[1]*y2
        if new_state[0]< -self.max_demands[0]:
            print('error 1')
        if new_state[1] < -self.max_demands[1]:
            print('error 2')
        return {'reward': profit, 'state':new_state}



            # Model Based Methods
    def get_probability(self,d, item_no):
        if item_no == 1:
            prob = poisson.pmf(d,self.mu1)
        else:
            prob = poisson.pmf(d,self.mu2)
        return prob

    def get_expected_expiration_cost(self,state,action):
        x1 = int(state[0])
        x2 = int(state[1])
        y1 = int(action[0])
        y2 = int(action[1])
        I1 = x1+y1
        I2 = x2+y2
        ### Case where both goods face expiration
        cost = 0
        total_prob = 0
        for d1 in range():
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
        I1 = x1+y1
        I2 = x2 + y2
        # Case where both goods face expiration
        revenue = 0
        cost = 0
        max1 = int(self.max_demands[0])
        max2 = int(self.max_demands[1])
        total_prob = 0
        for d1 in range(I1 + 1):
            for d2 in range(I2 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[0] * max(x1 - d1, 0) + self.theta[1] * max(x2, d2,0) * probability  # expiration cost
                cost += (self.h[0] * min(y1, I1 - d1) + self.h[1] * min(y2, I2 - d1)) * probability  # holding cost
                revenue += (d1 * self.p[0] + d2 * self.p[1]) * probability

        for d1 in range(I1 + 1):
            for d2 in range(I2 + 1, I2 + int((I1 - d1) / self.gamma1) + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[0] * max(x1 - int(d1 - self.gamma1 * (d2 - I2)), 0) * probability  # expiration cost
                cost += self.h[0] * min(y1, I1 - d1) * probability  # holding cost
                cost += self.b[1] * int((1 - self.gamma1) * (d2 - I2)) * probability  # backorder cost
                revenue += (d1 * self.p[0] + I2 * self.p[1] + int(d1 - self.gamma1 * (d2 - I2)) * self.p[1]) * probability

        for d1 in range(I1 + 1):
            for d2 in range(I2 + int((I1 - d1) / self.gamma1) + 1, max2 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.b[1] * int(d2 + d1 - I2 - I1) * probability  # backorder cost
                revenue += (d1 * self.p[0] + (I1 + I2 - d1) * self.p[1]) * probability

        for d2 in range(I2 + 1):
            for d1 in range(I1 + 1, I1 + int((I2 - d2) / self.gamma2) + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.theta[1] * min(0, x2 - d2) * probability  # expiration cost
                cost += self.h[1] * (I2 - d2 - int(self.gamma2 * (d1 - I1))) * probability  # holding cost
                cost += self.b[0] * int((1 - self.gamma2) * (d1 - I1)) * probability
                revenue += (I1 * self.p[0] + d2 * self.p[1] + self.p[0] * int(self.gamma2 * (d1 - I1))) * probability

        for d2 in range(I2 + 1):
            for d1 in range(I1 + int((I2 - d2) / self.gamma2) + 1, max1 + 1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += self.b[0] * int(d1 + d2 - I1 - I2) * probability  # backorder cost
                revenue += (I1 * self.p[0] + d2 * self.p[1] + (I2 + I1 - d2) * self.p[0]) * probability

        for d1 in range(I1 + 1, max1+1):
            for d2 in range(I2 + 1, max2+1):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                cost += (self.b[0] * (d1 - I1) + self.b[1] * (d2 - I2)) * probability  # backorder cost
                revenue += (self.p[0] * I1 + self.p[1] * I2) * probability

        return revenue-cost

    def calculate_expected_profits(self):
        for x1 in self.inv_range[0]:
            for x2 in tqdm(range(-int(self.max_demands[1]),int(self.max_demands[1]))):
                for y1 in range(max(0,-int(x1)),int(self.max_demands[0]-x1+1)):
                    for y2 in range(max(0,-int(x2)),int(self.max_demands[1] - x2 + 1)):
                        index = (y1, y2, int(x1+self.max_demands[0]+1), int(x2+self.max_demands[1]+1))
                        self.expected_profit[index] = -y1*self.c[0]-y2*self.c[0] +\
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
            for d2 in range(x2 + y2 + 1, min(x2 + y2 + 1 + int((x1 - d1) / self.gamma1),int(self.max_demands[1]+1))):
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
            for d1 in range(x1 + y1 + 1, min(x1 + y1 + 1 + int((x2 - d2 / self.gamma2)),int(self.max_demands[0]+1))):
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
            for d2 in range(x2 + y2 + 1, min(x2 + y2 + 1 + int((x1 + y1 - d1) / self.gamma1),int(self.max_demands[1]+1))):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [int(x1 + y1 - d1 - self.gamma1 * (d2 - x2 - y2)), int((1 - self.gamma1) * (
                            d2 - x2 - y2))]
                total_prob += probability
                if probability >0:
                    revenue += value_table[tuple(index)] * probability

        # case where 1 is subbed for 2 and all units are used
        for d1 in range(x1, x1 + y1 + 1):
            for d2 in range(x2 + y2 + 1 + int((x1 + y1 - d1) / self.gamma1), int(self.max_demands[1]+1)):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [0, int(d2 + d1 - x2 - y2 - x1 - y1)]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # case where 2 is subbed for 1
        for d2 in range(x2, x2 + y2 + 1):
            for d1 in range(x1 + y1 + 1, min(x1 + y1 + 1 + int((x2 + y2 - d2) / self.gamma2),int(self.max_demands[0]+1))):
                probability = self.get_probability(d1, 1) * self.get_probability(d2, 2)
                index = [int((1 - self.gamma2) * (d1 - x1 - y1)),int(x2 + y2 + self.gamma2 * (d1 - x1 - y1))]
                total_prob += probability
                if probability > 0:
                    revenue += value_table[tuple(index)] * probability

        # Case where 2 is subbed for 1 and all units are sold
        for d2 in range(x2, x2 + y2 + 1):
            for d1 in range(x1 + y1 + 1 + int((x2 + y2 - d2) / self.gamma2), int(self.max_demands[0]+1)):
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



