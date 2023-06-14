import matplotlib.pyplot as plt
import numpy as np

def plot_loss(n):
    y = []
    for i in range(1,n):
        enc = np.load('output/epoch_loss_UCF-Crime:{}.npy'.format(i))
        # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
        tempy = list(enc)
        if i>1:
            tempy = [x for x in tempy if x < 2]
        y += tempy
    x = range(0,len(y))
    plt.plot(x, y, color='b')
    plt_title = 'UCF-Crime Loss\nbatch size = 2; learning rate:1e-6'
    # plt.legend('batch size = 2; learning rate:1e-6')
    plt.title(plt_title)
    plt.xlabel('per 20 batches')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()

if __name__ == "__main__":
    plot_loss(8)