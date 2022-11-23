import numpy as np
import torch
import os
import time
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

epochs = 50
def plot_estimate(pdf):

    for epoch in range(epochs):
        data = np.load("./checkpoint/EST_E_"+str(epoch+1)+".npy")
        num = data.shape[0]
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            eigen = data[i,0,:].reshape(1,100)
            weight = data[i,1,:].reshape(1,100)
            fig.add_subplot(1, num, i+1).set_title('Epoch '+str(epoch)+' Layer ' + str(i+1))
            get_esd_plot(eigen, weight)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return num


def plot_formula(pdf, num):

    for epoch in range(epochs):
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            hessian = np.load( "./checkpoint/FOR_L_"+str(i+1)+"_E_"+str(epoch+1)+".npy")

            ax = fig.add_subplot(1, num, i+1)
            plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))

            hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
            weights = np.ones_like(hessian)/float(len(hessian))
            plt.hist(hessian, bins = 3000, weights=weights)
            ax.set_xscale('log')
            plt.xlim(left=1e-5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()            
    return

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    with PdfPages('./graph/plot_Estimate.pdf') as pdf:
        num = plot_estimate(pdf)
    with PdfPages('./graph/plot_Formula.pdf') as pdf:
        plot_formula(pdf, num)



