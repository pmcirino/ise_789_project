import numpy as np
from algorithms.single_product.model_based import PolicyIteration
import environments.infinte_demand.single_item as single_poisson
import environments.infinte_demand.two_item as two_item_poisson

from algorithms.two_product.temporal_difference import TemporalDifference
from algorithms.two_product.monte_carlo import PolicyIteration as McPolicyIteration
from output_files.save_outputs import save_results
demand_rate = [5, 6]
max_demand = 0.99
h = np.array([1,1])
b = np.array([1,1])
p = np.array([15,15])
c = np.array([4, 4])
theta = np.array([8,2])
gammas = np.array([.4, .8])
w_2= 9
two_product = two_item_poisson.TwoItem(demand_rates=demand_rate,max_demand=max_demand,
                                      h=h,b=b,p=p,c=c,theta=theta,substitution_rates=gammas)

#get_two = single_poisson.SingleItem(demand_rate=demand_rate[1],max_demand=max_demand,h=h[1],
                                   # b=b[1],p=p[1],c=c[1],theta=theta[1])
#inst = PolicyIteration(get_two,0.01,0.8, 'poisson_w2_run')
#test = inst.policy_optimization()

# Two product Q Learning w2 #
td = TemporalDifference(two_product, 0.0001, 0.8, 0.9999,10000000, 'poisson_q_learning_reaction', w2=9)
td.q_learning()
# Two Product Regular #
td = TemporalDifference(two_product, 0.0001, 0.8, 0.9999,10000000, 'poisson_q_learning')
td.q_learning()

# Two product MC w2 #
#mc = McPolicyIteration(two_product,100000,100,0.999,0.8,name='poisson_mc_pi_reaction',w2=4)
#mc.policy_iteration()
# Two Product MC #
#mc = McPolicyIteration(two_product,100000,100,0.999,0.8,name='poisson_mc_pi')
#mc.policy_iteration()
