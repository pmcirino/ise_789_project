import numpy as np
import matplotlib.pyplot as plt
import os
files = ['poisson_mc_pi_reaction_values.npy','poisson_mc_pi_values.npy',
         'poisson_q_learning_reaction_values.npy','poisson_q_learning_values.npy']
plot_titles = ['Monte-Carlo PI with $w_2=9$','Monte-Carlo PI','Q-Learning $w_2=9$','Q-Learning']
output_files =['monte_carlo_values2.png','q_learning_values2.png']
Values = []
dest = '/Users/paul/Documents/789_Project/ise_789_project/output_files/values/'
for file in files:
    path_name = dest+file
    path = os.path.abspath(path_name)
    tmp = np.load(path)
    Values.append(tmp)
x1_range = Values[2][1:,0]
x2_range = Values[2][0,1:]
print(len(x2_range)//4+1)
for i in range(2):
    fig, axs = plt.subplots(nrows=len(x2_range)//4+1, ncols=2, figsize =(14,10))
    for x2 in range(len(x2_range)//2+2,len(x2_range)+1):
        index = x2 % (len(x2_range)//4+1)
        print(index)
        if x2 <= 3*len(x2_range)//4+2:
            title = f'$x_1={x2-x2_range[-1]}$'
            axs[index, 0].plot(x1_range,Values[2*i][1:,x2], label=plot_titles[2*i])
            axs[index, 0].plot(x1_range, Values[2*i+1][1:, x2], label=plot_titles[2*i+1])
            axs[index, 0].annotate(title, xy=(0,0.5),xytext=(-axs[index,0].yaxis.labelpad - 5, 0),
                    xycoords=axs[index,0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        else:
            title = f'$x_1={x2-x2_range[-1]}$'
            axs[index,1].plot(x1_range,Values[2*i][1:,x2],label=plot_titles[2*i])
            axs[index, 1].plot(x1_range, Values[2*i+1][1:, x2],label=plot_titles[2*i+1])
            axs[index,1].annotate(title, xy=(1,0.5),xytext=(-axs[index,1].yaxis.labelpad - 5, 0),
                    xycoords=axs[index,1].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    axs[-1, 0].set_xlabel('$x_1$')
    axs[-1, 1].set_xlabel('$x_1$')
    axs[-1, 0].set_xticks(x1_range)
    axs[-1, 1].set_xticks(x1_range)
    axs[-2,1].legend()
    fig.tight_layout(pad=3.0)
    path = '/Users/paul/Documents/789_Project/ise_789_project/output_files/plots_images/poisson_value_plots/'+output_files[i]
    fig.savefig(os.path.abspath(path), dpi=100)

fig, axs = plt.subplots(nrows=len(x2_range)//4+1, ncols=2, figsize =(14,10))
for x2 in range(len(x2_range)//2+1,len(x2_range)+1):
    index = x2 % (len(x2_range)//4+1)
    if x2 <= 3*len(x2_range)//4:
        title = f'$x_1={x2-x2_range[-1]}$'
        axs[index, 0].plot(x1_range,Values[1][1:,x2], label=plot_titles[1])
        axs[index, 0].plot(x1_range, Values[3][1:, x2], label=plot_titles[3])
        axs[index, 0].annotate(title, xy=(0,0.5),xytext=(-axs[index,0].yaxis.labelpad - 5, 0),
                xycoords=axs[index,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    else:
        title = f'$x_1={x2-x2_range[-1]}$'
        axs[index, 1].plot(x1_range,Values[1][1:,x2], label=plot_titles[1])
        axs[index, 1].plot(x1_range, Values[3][1:, x2], label=plot_titles[3])
        axs[index,1].annotate(title, xy=(1,0.5),xytext=(-axs[index,1].yaxis.labelpad - 5, 0),
                xycoords=axs[index,1].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
axs[-1, 0].set_xlabel('$x_1$')
axs[-1, 1].set_xlabel('$x_1$')
axs[-1, 0].set_xticks(x1_range)
axs[-1, 1].set_xticks(x1_range)
axs[-2,1].legend()
fig.tight_layout(pad=3.0)
path = '/Users/paul/Documents/789_Project/ise_789_project/output_files/plots_images/poisson_value_plots/comparison_plot2.png'
fig.savefig(os.path.abspath(path), dpi=100)

plt.show()