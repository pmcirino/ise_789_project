from plots.action_plots import TwoProduct
import matplotlib.pyplot as plt
from output_files.save_outputs import open_actions
import os
files = ['indepdent_two_item_pi_actions.npy',
         'mc_pi_actions.npy',
       'mc_pi_reaction_actions.npy',
       'mc_policy_actions.npy',
       'poisson_pi_actions.npy',
       'q_learning_actions.npy',
       'q_learning_reaction_actions.npy',
       'q_learning_test_actions.npy',
       'reaction_two_item_pi_actions.npy',
       'two_item_pi_actions.npy']

dest = '/Users/paul/Documents/789_Project/ise_789_project/output_files/plots_images/finite_action_plots/'
for file in files:
    path_name = dest+file
    path = os.path.splitext(path_name)[0]+'.png'
    c = TwoProduct(file)
    tmp = c.create_plot()
    tmp.savefig(path,dpi=100)
