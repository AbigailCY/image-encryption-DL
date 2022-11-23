import numpy as np
import torch
import os
import time
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

epochs = 50
def plot_estimate_ylog(pdf):

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

def plot_estimate_1(pdf):

    for epoch in range(epochs):
        data = np.load("./checkpoint/EST_E_"+str(epoch+1)+".npy")
        num = data.shape[0]
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            eigen = data[i,0,:].reshape(100)[::-1]
            weights = np.ones_like(eigen)/float(len(eigen))

            ax = fig.add_subplot(1, num, i+1)
            plt.hist(eigen, bins = 100, weights=weights)
            plt.title('Epoch '+str(epoch)+' Layer ' + str(i+1))
            # ax.set_yscale('log')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return num

def plot_estimate_bothlog(pdf):
    for epoch in range(epochs):
        data = np.load("./checkpoint/EST_E_"+str(epoch+1)+".npy")
        num = data.shape[0]
        fig = plt.figure(figsize=(5*num,4))
        for i in range(num):
            eigen = data[i,0,:].reshape(1,100)
            weight = data[i,1,:].reshape(1,100)
            fig.add_subplot(1, num, i+1).set_title('Epoch '+str(epoch)+' Layer ' + str(i+1))
            get_esd_plot_both(eigen, weight)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return num

def plot_estimate_xlog(pdf):
    for epoch in range(epochs):
        data = np.load("./checkpoint/EST_E_"+str(epoch+1)+".npy")
        num = data.shape[0]
        fig = plt.figure(figsize=(5*num,4))
        for i in range(num):
            eigen = data[i,0,:].reshape(1,100)
            weight = data[i,1,:].reshape(1,100)
            fig.add_subplot(1, num, i+1).set_title('Epoch '+str(epoch)+' Layer ' + str(i+1))
            get_esd_plot_x(eigen, weight)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return num

def plot_formula_xlog(pdf, num):

    for epoch in range(epochs):
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            hessian = np.load( "./checkpoint/FOR_L_"+str(i+1)+"_E_"+str(epoch+1)+".npy")

            ax = fig.add_subplot(1, num, i+1)
            plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))

            hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
            weights = np.ones_like(hessian)/float(len(hessian))
            plt.hist(hessian, bins = 1000, weights=weights)
            ax.set_xscale('log')
            plt.xlim(left=1e-5)

        plt.tight_layout()
        pdf.savefig()
        plt.close() 
        print("xlog epoch "+str(epoch)+" finished")           
    return

def plot_formula_ylog(pdf, num):

    for epoch in range(epochs):
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            hessian = np.load( "./checkpoint/FOR_L_"+str(i+1)+"_E_"+str(epoch+1)+".npy")

            ax = fig.add_subplot(1, num, i+1)
            plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))

            hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
            weights = np.ones_like(hessian)/float(len(hessian))
            plt.hist(hessian, bins = 1000, weights=weights)
            ax.set_yscale('log')
            plt.xlim(left=1e-5)

        plt.tight_layout()
        pdf.savefig()
        plt.close() 
        print("ylog epoch "+str(epoch)+" finished")              
    return

def plot_formula_bothlog(pdf, num):

    for epoch in range(epochs):
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            hessian = np.load( "./checkpoint/FOR_L_"+str(i+1)+"_E_"+str(epoch+1)+".npy")

            ax = fig.add_subplot(1, num, i+1)
            plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))

            hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
            weights = np.ones_like(hessian)/float(len(hessian))
            plt.hist(hessian, bins = 1000, weights=weights)
            ax.set_xscale('log')
            ax.set_yscale('log')
            plt.xlim(left=1e-5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("both epoch "+str(epoch)+" finished")               
    return

def plot_formula_nolog(pdf, num):

    for epoch in range(epochs):
        fig = plt.figure(figsize=(5*num,4))

        for i in range(num):
            hessian = np.load( "./checkpoint/FOR_L_"+str(i+1)+"_E_"+str(epoch+1)+".npy")

            ax = fig.add_subplot(1, num, i+1)
            plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))

            hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
            weights = np.ones_like(hessian)/float(len(hessian))
            plt.hist(hessian, bins = 1000, weights=weights)
            plt.xlim(left=1e-5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        print("nolog epoch "+str(epoch)+" finished")            
    return

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    t = time.time()
    with PdfPages('./graph/plot_Estimate_new.pdf') as pdf:
        num = plot_estimate_1(pdf)

    print(time.time()-t)
    print("Estimate y finished")
    # t = time.time()

    # t = time.time()
    # with PdfPages('./graph/plot_Estimate_ylog.pdf') as pdf:
    #     num = plot_estimate_ylog(pdf)

    # print(time.time()-t)
    # print("Estimate y finished")
    # t = time.time()

    # with PdfPages('./graph/plot_Estimate_bothlog.pdf') as pdf:
    #     num = plot_estimate_bothlog(pdf)

    # print(time.time()-t)
    # print("Estimate both finished")
    # t = time.time()

    # with PdfPages('./graph/plot_Estimate_xlog.pdf') as pdf:
    #     num = plot_estimate_xlog(pdf)

    # print(time.time()-t)
    # print("Estimate x finished")
    # t = time.time()


    # with PdfPages('./graph/plot_Formula_xlog.pdf') as pdf:
    #     plot_formula_xlog(pdf, num)
    
    # print(time.time()-t)
    # print("Formula x finished")
    # t = time.time()

    # with PdfPages('./graph/plot_Formula_ylog.pdf') as pdf:
    #     plot_formula_ylog(pdf, num)
    
    # print(time.time()-t)
    # print("Formula y finished")
    # t = time.time()

    # with PdfPages('./graph/plot_Formula_bothlog.pdf') as pdf:
    #     plot_formula_bothlog(pdf, num)
    
    # print(time.time()-t)
    # print("Formula both finished")
    # t = time.time()

    # with PdfPages('./graph/plot_Formula_nolog.pdf') as pdf:
    #     plot_formula_nolog(pdf, num)
    
    # print(time.time()-t)
    # print("Formula no finished")


