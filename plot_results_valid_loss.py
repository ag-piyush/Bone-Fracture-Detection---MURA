import matplotlib.pylab as plt
import json
import numpy as np


def plot_MURA(save=True):

    with open("./log/experiment_log_MURA.json", "r") as f:
        d = json.load(f)


    valid_loss = 100 * np.array(d["valid_loss"])[:, 0]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')                                                   
    ax1.plot(valid_loss, color="tomato", linewidth=1, label='valid_loss')
    
    ax1.legend(loc=0)
    

	
    ax1.grid(True)

    if save:
        fig.savefig('./figures/plot_MURA_valid_loss.jpg')

    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    plot_MURA()
