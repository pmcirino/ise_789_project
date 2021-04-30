import numpy as np
import environments.finite_demand.single_item as single_item_finite
import environments.finite_demand.two_item as two_item_finite
import environments.infinte_demand.single_item as single_poisson
import environments.infinte_demand.two_item as two_item_poisson
from algorithms.single_product.model_based import PolicyIteration
from algorithms.two_product.model_based import PolicyIteration as TwoPolicyIteration
from algorithms.two_product.temporal_difference import TemporalDifference
from algorithms.two_product.monte_carlo import PolicyIteration as McPolicyIteration
from output_files.save_outputs import save_results


DEMAND_VALUES = np.array([np.arange(8), np.array([0,1,2,3,4])], dtype=object)
DEMAND_PROB = np.array([np.array([1/8]*8),np.array([0.2]*5)], dtype=object)
h = np.array([1,1])
b = np.array([1,1])
p = np.array([15,15])
c = np.array([4, 4])
theta = np.array([8,2])
gammas = np.array([.4, .8])

# Single Product #
item1 = single_item_finite.SingleItem(DEMAND_VALUES[0],DEMAND_PROB[0],h[0],b[0],p[0],c[0],theta[0])
item2 = single_item_finite.SingleItem(DEMAND_VALUES[1],DEMAND_PROB[1],h[1],b[1],p[1],c[1],theta[1])
inst1 = PolicyIteration(item1,0.01,0.8,'item1')
inst2 = PolicyIteration(item2,0.01,0.8,'item2')
inst1.policy_optimization()
inst2.policy_optimization()
actions1 = np.load('./output_files/actions/item1_actions.npy')
actions2 = np.load('./output_files/actions/item2_actions.npy')
x1_max = int(np.nanmax(actions1[:,0]))
x2_max = int(np.nanmax(actions2[:,0]))
policy = np.zeros([2]+[len(actions1[:,0]),len(actions2[:,0])])
for x1 in range(0,int(2*x1_max+1)):
    for x2 in range(0,int(2*x2_max+1)):
        policy[0,x1,x2] = actions1[x1,1]
        policy[1, x1, x2] = actions2[x2,1]
two_product = two_item_finite.TwoItem(demand_values=DEMAND_VALUES,demand_probabilities=DEMAND_PROB,
                                      h=h,b=b,p=p,c=c,theta=theta,substitution_rates=gammas)
inst = TwoPolicyIteration(two_product,0.01,0.8,'reaction_two_item_pi')
values = inst.policy_evaluation(policy)
save_results('indepdent_two_item_pi',{'values':values,'policy':policy})

# Two Product with Base Stock
inst.reaction_policy_iteration(w2=4)
# Two prdocut joint solution
inst.file_name = 'two_item_pi'
inst.policy_optimization()

# Two product Q Learning w2 #
#td = TemporalDifference(two_product, 0.01, 0.8, 0.999,1000000, 'q_learning_reaction', w2=4)
#td.q_learning()
# Two Product Regular #
#td = TemporalDifference(two_product, 0.01, 0.8, 0.999,1000000, 'q_learning')
#td.q_learning()

# Two product MC w2 #
#mc = McPolicyIteration(two_product,100000,10,0.99,0.8,name='mc_pi_reaction',w2=4)
#mc.policy_iteration()
# Two Product MC #
#mc = McPolicyIteration(two_product,100000,10,0.99,0.8,name='mc_pi')
#mc.policy_iteration()
