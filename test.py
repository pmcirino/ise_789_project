from output_files.save_outputs import open_actions
from plots.action_plots import TwoProduct
import matplotlib.pyplot as plt
a = open_actions('poisson_q_learning_actions.npy')

fig = TwoProduct('poisson_q_learning_actions.npy')
fig.create_plot()
plt.show()
