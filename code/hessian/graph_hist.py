import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def line(eigenvalues, num_bins=1000, sigma_squared=1e-5):
    # plt.plot(grids, density, linestyle = '-', linewidth=0.5)
    # plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density', fontsize=8, labelpad=6)
    plt.xlabel('Eigenvlaue', fontsize=8, labelpad=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    # plt.axis([lambda_min-overhead, lambda_max+overhead, None, None])



def plot_hist(net):
    print('Start '+net)
    fc1 = np.load( './hessian_eigen/'+net+'_fc1.npy' )
    fc2 = np.load( './hessian_eigen/'+net+'_fc2.npy' )
    fc3 = np.load( './hessian_eigen/'+net+'_fc3.npy' )
    epochs = fc1.shape[0]
    fc1, fc2, fc3 = fc1.reshape(epochs, -1), fc2.reshape(epochs, -1), fc3.reshape(epochs, -1)

    with PdfPages('./graph/'+net+'_hist.pdf') as pdf: 

        for epoch in range(1, epochs+1):
            plt.figure(figsize=(11,3))

            plt.subplot(131)
            plt.hist(fc1[epoch-1])
            # sns.distplot(fc1, norm_hist=True,kde=False)
            plt.subplot(131).set_title('Eopch: '+ str(epoch)+' Layer FC1')

            plt.subplot(132)
            plt.hist(fc2)
            # sns.distplot(fc2, norm_hist=True,kde=False)
            plt.subplot(132).set_title('Eopch: '+ str(epoch)+' Layer FC2')

            plt.subplot(133)
            plt.hist(fc3)
            plt.subplot(133).set_title('Eopch: '+ str(epoch)+' Layer FC3')

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished')

    print(net+' finished.')

if __name__ == '__main__':
    plot_hist('LM')
    # plot_hist('AM')
    # plot_hist('LC')
    # plot_hist('AC')
