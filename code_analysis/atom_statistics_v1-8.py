'''

Version 1.6 

Modifications: Bruno Ximenez
Date: 08.02.2021

This code is intended to be the front end 
of the general software to run the most frequent 
kind of analysis done in the lab with the data from the camera. 

'''

import matplotlib.pyplot as plt
import import_data as imp_data
import plot_trap_site as plt_trap
import config
import sys
import numpy as np
import temperature_single_atom as sim


'''
Here get Boolean data.
Sequence of zeros and ones
corresponding to trap site occupied or not. That simple.
'''
x_values, initial_bool, assembled_bool, assembled_list_bool, recapture_bool, atom_loss_bool = imp_data.get_data()


# Plot Histogram with number of atoms in each run
plt_trap.atom_number(initial_bool)

'''
Plot occupation probability for each trap site.
'''
plt_trap.init_plot(initial_bool)

calc_diff = False
if calc_diff == True:
    # If True will plot image on 'assembled bool - initial'
    plt_trap.diff_plot(initial_bool, assembled_bool)


# Filter data for rearrangement ??
filter = False
if filter == False:
    plt_trap.reassembled_plot(assembled_bool)
else:
    filtered_data_reass, assembled_bool_ps = plt_trap.filter_data_plot(
        initial_bool, assembled_bool)


# Plot defects ???
defects = False
if defects == True:
    # number of defects above which the run is considered as failed
    config.n_defect_thresh = 12
    plt_trap.defects(assembled_bool)


# Plot recapture and simulate?
simulate = True
if simulate == True:
    sim.run(temp=60e-6)
    plt_trap.calc_recap(x_values, assembled_bool, recapture_bool)


plt.grid()
plt.show()
