import matplotlib.pyplot as plt
import numpy as np
import os
import stat_functions as stat
import config
import scipy.special as sc
import sys

fontsize = 14
'''
Some plot settings
'''
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

def plot(data_bool, size_array, title, fig_name):
    print('\n', title, '\n', data_bool)
    data_to_plot = np.rot90(stat.get_init_filling(data_bool)[
                            0].reshape(config.size_array, config.size_array))
    stat.plot_heatmap(data_to_plot, stat.get_init_filling(
        data_bool)[1], config.limits_1, title)
    par_dir = os.path.dirname(os.getcwd())
    base_name = os.path.basename(os.path.normpath(par_dir))
    file_name = os.path.join(par_dir, base_name + fig_name)
    plt.savefig(file_name, bbox_inches='tight')


def init_plot(initial_bool):
    # print('\nAssembled bool Post-process \n', assembled_bool_ps)

    print('\nPlotting initial distribution of atoms in each trap site...')
    plot(initial_bool, config.size_array, 'Initial data to plot', '_init.png')


def diff_plot(bool_1, bool_2):
    # print('\nAssembled bool Post-process \n', assembled_bool_ps)

    print('\nPlotting missing atoms Probability in each trap site...')
    bool_diff = bool_2 - bool_1
    plot(bool_diff, config.size_array, 'Differential data to plot', '_diff.png')


def reassembled_plot(reassembled_bool):
    print('\nPlotting reassembled Probability data for each site')
    plot(reassembled_bool, config.size_array,
         'reassembled data to plot', '_reass.png')


def filter_data_plot(initial_bool, assembled_bool):
    assembled_bool_ps = stat.get_assemble_filling_postselected(
        initial_bool, assembled_bool, config.selectedROI_list, config.n_defect_thresh)[3]
    reass_data_to_plot = np.rot90(stat.get_assemble_filling_postselected(
        initial_bool, assembled_bool, config.selectedROI_list, config.n_defect_thresh)[0].reshape(config.size_array, config.size_array))
    print("Rearranged filling")
    stat.plot_heatmap(reass_data_to_plot, stat.get_assemble_filling_postselected(initial_bool, assembled_bool,
                                                                                 config.selectedROI_list, config.n_defect_thresh)[1], main_res_str='Averaged rearr. proba per trap')
    par_dir = os.path.dirname(os.getcwd())
    base_name = os.path.basename(os.path.normpath(par_dir))
    file_name = os.path.join(par_dir, base_name + '_reass.png')
    plt.savefig(file_name, bbox_inches='tight')
    return reass_data_to_plot, assembled_bool_ps


def defects(assembled_bool):
    plt.figure()
    n_run = assembled_bool.shape[0]  # number of runs

    # Calculate number of defect inside the final array
    defect = []
    for k in range(n_run):
        defect.append(len(config.selectedROI_list) -
                      np.sum(assembled_bool[k, config.selectedROI_list]))

    print(np.max(defect))
    n_per_bin, bins, patches = plt.hist(x=defect, bins=np.max(defect), color='#0504aa',
                                        alpha=1, rwidth=1, label='data')

    # plt.figure('Defects')
    plt.grid(axis='y', alpha=0.3)
    plt.xlabel('Number of defect', fontsize=fontsize)
    plt.ylabel('Occurence', fontsize=fontsize)
    plt.title('Defect after rearrangement')
    plt.xticks(np.arange(0, np.max(defect)+1, step=np.floor(np.max(defect)/4)))
    maxfreq = n_per_bin.max()
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) # Set a clean upper y-axis limit.

    # Binomial law
    def defect_occ(p, N_move, k):
        return sc.binom(N_move, k)*p**k*(1-p)**(N_move-k)

    k_list = np.arange(0, len(bins))
    # print(k_list)

    plt.plot(k_list+0.5, n_run*defect_occ(0.053, 32, k_list),
             '-ro', label='Binomial fit: p=0.053, N=32')
    #plt.plot(k_list+0.5,'-ro',label='Binomial fit: p=0.04, N=20')
    plt.legend()
    par_dir = os.path.dirname(os.getcwd())
    base_name = os.path.basename(os.path.normpath(par_dir))
    file_name = os.path.join(par_dir, base_name + '_defect_statistics.png')
    plt.savefig(file_name, bbox_inches='tight')

    # In[ ]:

    trap_nb = assembled_bool.shape[1]
    print('Total number of runs:', int(np.sum(n_per_bin[0:len(bins)])))
    print('Probability to have a full array:',
          n_per_bin[0]/np.sum(n_per_bin[:len(bins)])*trap_nb, '%')
    print('Probability to have a full array with failed runs removed:',
          n_per_bin[0]/np.sum(n_per_bin[:config.n_defect_thresh])*trap_nb, '%')


def atom_err(n):
    err = np.sqrt(np.sum(n)) / len(n)
    return err


def atom_number(initial_bool):
    fontsize = 14

    n_run = initial_bool.shape[0]
    atom_each_run = np.sum(initial_bool, axis=1)
    avg = np.sum(atom_each_run) / n_run
    err_n = atom_err(atom_each_run)
    print('\nAverage number of atoms trapped per run: ', avg)
    print('\nStd Dev loading: ', np.std(atom_each_run))
    print('\nshotnoise limit: ', err_n)
    print('\nNumber of runs: ', n_run)

    plt.figure('Atom loading distribution')

    plt.hist(atom_each_run, bins='auto', color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.xlabel('Number fo atoms trapped', fontsize=fontsize)
    plt.ylabel('Events', fontsize=fontsize)
    par_dir = os.path.dirname(os.getcwd())
    base_name = os.path.basename(os.path.normpath(par_dir))
    file_name = os.path.join(par_dir, base_name + '_hist_atoms.png')
    plt.savefig(file_name, bbox_inches='tight')


def proba_calc_err(p, a, err_a, b, err_b):
    err = (err_a / a) ** 2 + (err_b / b) ** 2
    err = p * np.sqrt(err)
    return err


def calc_loop(scan_values, bools):
    x_unique = np.unique(scan_values)
    print('Number of repetitions in each loop iteration: ')
    counter = 0
    n_atom = np.array([])
    n_atom_err = np.array([])
    for x in x_unique:
        current_run_list = np.argwhere(scan_values == x)
        n_rep_each_iteration = np.size(current_run_list)
        tmp = np.array([])
        for i in range(n_rep_each_iteration):
            sum_atom = np.sum(bools[i + counter,:])
            tmp = np.append(tmp, sum_atom)
        
        n_atom = np.append(n_atom, np.average(tmp))
        print('len', len(tmp))
        n_atom_err = np.append(n_atom_err, np.sqrt( np.average(tmp) / len(tmp) ) )
        counter += n_rep_each_iteration
    
    return n_atom, n_atom_err


def calc_recap(x_values, assembled_bool, recapture_bool):
    n_atom, n_atom_std = calc_loop(x_values, assembled_bool)
    n_recaptured, n_recaptured_std = calc_loop(x_values, recapture_bool)

    prob_recap = n_recaptured / n_atom
    prob_recap_err = proba_calc_err(prob_recap, n_atom, n_atom_std, n_recaptured, n_recaptured_std)

    # plt.plot(np.unique(x_values), prob_recap, 'o', color='orangered', label='Experimental data')
    plt.errorbar(np.unique(x_values), prob_recap, yerr=prob_recap_err, capthick=2, fmt='o', color='orangered', label='Experimental data')
    plt.xlabel('Release time (us)', fontsize=fontsize)
    plt.ylabel('Recapture probability', fontsize=fontsize)
    plt.ylim(0,1)
    plt.legend()
    # plt.show()



