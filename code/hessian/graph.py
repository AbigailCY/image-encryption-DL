import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def line(eigenvalues, num_bins=1000, sigma_squared=1e-5):

    e_max = np.max(eigenvalues)
    e_min = np.min(eigenvalues)
    overhead = (np.max(eigenvalues) - np.min(eigenvalues))/15
    lambda_max = e_max + overhead
    lambda_min = e_min - overhead
    print(e_max, e_min)

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    # sigma = sigma_squared * max(1, (lambda_max - lambda_min))
    sigma = sigma_squared * (lambda_max - lambda_min)

    density_output = np.zeros(num_bins)
    
    for j in range(num_bins):
        x = grids[j]
        tmp_result = gaussian(eigenvalues, x, sigma)
        density_output[j] = np.mean(tmp_result)
    density = density_output
    # print(density.shape)
    # normalization = np.sum(density) * (grids[1] - grids[0])
    # density = density / normalization

    plt.plot(grids, density, linestyle = '-', linewidth=0.5)
    # plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density', fontsize=8, labelpad=6)
    plt.xlabel('Eigenvlaue', fontsize=8, labelpad=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis([lambda_min-overhead, lambda_max+overhead, None, None])

def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

def plot_line(net):
    print('Start '+net)
    fc1 = np.load( './hessian_eigen/'+net+'_fc1.npy' )
    fc2 = np.load( './hessian_eigen/'+net+'_fc2.npy' )
    fc3 = np.load( './hessian_eigen/'+net+'_fc3.npy' )
    epochs = fc1.shape[0]
    fc1, fc2, fc3 = fc1.view(epochs, -1), fc2.view(epochs, -1), fc3.view(epochs, -1)

    with PdfPages('./graph'+net+'_line.pdf') as pdf: 

        for epoch in epochs:
            plt.figure(figsize=(11,3))

            plt.subplot(131)
            line(fc1)
            plt.subplot(131).set_title('Eopch: '+ str(epoch)+' Layer FC1')

            plt.subplot(132)
            line(fc2)
            plt.subplot(132).set_title('Eopch: '+ str(epoch)+' Layer FC2')

            plt.subplot(133)
            line(fc3)
            plt.subplot(133).set_title('Eopch: '+ str(epoch)+' Layer FC3')

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished')

    print(net+' finished.')

if __name__ == '__main__':
    # plot_line('LM')
    # plot_line('AM')
    # plot_line('LC')
    # plot_line('AC')

    # fc1 = np.load( './hessian_eigen/'+net+'_fc1.npy' )
    # epochs = fc1.shape[0]
    # fc1 = fc1.view(epochs, -1)