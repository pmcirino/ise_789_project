from plots.action_plots import TwoProduct
import matplotlib.pyplot as plt
from output_files.save_outputs import open_actions
import os
files = ['poisson_mc_pi_actions.npy',
'poisson_mc_pi_reaction_actions.npy',
'poisson_q_learning_actions.npy',
'poisson_q_learning_reaction_actions.npy']
dest = '/Users/paul/Documents/789_Project/ise_789_project/output_files/plots_images/poisson_action_plots/'
for file in files:
    path_name = dest+file
    path = os.path.splitext(path_name)[0]+'.png'
    c = TwoProduct(file)
    tmp = c.create_plot()
    tmp.savefig(path,dpi=100)
