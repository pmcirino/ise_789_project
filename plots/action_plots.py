import matplotlib.pyplot as plt
import numpy as np
from output_files import save_outputs

class TwoProduct():
    def __init__(self, file_name):
        self.actions = save_outputs.open_actions(file_name)
        self.x1_range = len(self.actions[0,:,0])
        self.x2_range = len(self.actions[0, 0, :])
        self.x1_ticks = np.arange(-self.x1_range//2+1, self.x1_range//2 + 1)
        self.x2_ticks = np.arange(-self.x2_range//2+1, self.x2_range//2 + 1)
        print(self.x1_ticks)
    def create_plot(self):
        fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(12, 9))
        ax[0].imshow(self.actions[0], cmap='cool', origin='lower')
        ax[0].set_xlabel('x2')
        ax[0].set_ylabel('X1')
        ax[0].set_title('Order Quantity 1 ($y_1$)')
        ax[0].set_xticks(np.arange(self.x2_range))
        ax[0].set_yticks(np.arange(self.x1_range))
        ax[0].set_xticklabels(self.x2_ticks)
        ax[0].set_yticklabels(self.x1_ticks)
        ax[1].imshow(self.actions[1], cmap='cool', origin='lower')
        ax[1].set_xlabel('x2')
        ax[1].set_ylabel('X1')
        ax[1].set_title('Order Quantity 2 ($y_2$)')
        ax[1].set_xticks(np.arange(self.x2_range))
        ax[1].set_yticks(np.arange(self.x1_range))
        ax[1].set_xticklabels(self.x2_ticks)
        ax[1].set_yticklabels(self.x1_ticks)
        for i in range(0, self.x1_range):
            for j in range(0, self.x2_range):
                text = ax[0].text(j, i, int(self.actions[0,i, j]),
                                ha='center', va='center')
                text = ax[1].text(j, i, int(self.actions[1,i, j]),
                                ha='center', va='center')

        return fig