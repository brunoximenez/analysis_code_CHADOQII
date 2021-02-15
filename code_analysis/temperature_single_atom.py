# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:10:24 2016

@author: thierry.lahaye

Modified: Bruno Ximenez, 10.02.2021
Calculation has been vectorized.
Functions are called in modules. This is supposed to be a frontend now.
"""


import numpy as np
import matplotlib.pyplot as plt
import lib_calc as calc
import sys


def run(temp):
    # Trap depth in mK
    u0 = 1.

    print('Initializing simulation...\n')

    # Calculate varying gate time or T?
    vary_time = True

    if vary_time == True:
        # temp_init, temp_end, steps
        n_curves = 1
        temp_array = np.linspace(temp, 80e-6, n_curves)
        for j, T in enumerate(temp_array):
            print('Calculating through the loop: ', j)
            free_flight, recap = calc.sim_vary_gate(T, t_init=0, t_end=60e-6, steps=20, u0=u0)
            calc.plot(free_flight, recap, T, vary_time, 'Simulation varying free flight time', 'Gate open (us)')


    else:
        gate_opened = 50e-6
        T_list, recap = calc.sim_vary_temp(gate_opened, T_init=0, T_end=100e-6, steps=50, u0=u0)
        calc.plot(T_list, recap, gate_opened, vary_time, 'Simulation varying free flight time', 'T(uK)')

    # plt.show()




if __name__ == '__main__':
    run(T=60e-6)
#from scipy import interpolate
#
#x = np.load('recap_list.npy')
#y = np.load('tempe_list.npy')
#ind_sort = np.argsort(x)
#x = x[ind_sort]
#y = y[ind_sort]*1e6
#tck = interpolate.splrep(x,y,s=5)
#
#
#xnew = np.arange(0.25, 0.92, 0.001)
#ynew = interpolate.splev(xnew,tck)   # use interpolation function returned by `interp1d`