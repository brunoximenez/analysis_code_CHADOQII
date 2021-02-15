import numpy as np
import os
import import_functions as imp
import config


def get_data():
    print('Loading data....\n')
    # ## Config

    threshold_global_fail = config.threshold_global_fail
    threshold_global_recap = config.threshold_global_recap

    # number of defects above which the run is considered as failed
    n_defect_thresh = config.n_defect_thresh

    filename = 'selectedROI.dat'
    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, filename))
    data_list = []

    with open(file_path) as f:
        # Get the header
        header = next(f)
        # Get the datas
        for line in f:
            # strip() removes the \n, split('\t') return a list whose elements where separated by a \t
            data_list.append(line.strip().split('\t'))
    f.close()

    config.selectedROI_list = np.asarray(
        [indexROI[0] for indexROI in data_list])
    config.selectedROI_list = [int(elt) for elt in config.selectedROI_list]
    N_assembled = len(config.selectedROI_list)

    # read the data from the file
    filename = 'data.dat'
    file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, filename))
    data = imp.parseFile(file_path)

    # Open and read the ROI position file
    number_of_rois, roi_positions = imp.openROIfile()

    # Format the data
    x_values = data[:, 0]
    initial_bool, assembled_bool, assembled_list_bool, recapture_bool = imp.formatData(
        data, number_of_rois, threshold_global_fail, threshold_global_recap)

    n_traps = initial_bool.shape[1]  # number of runs
    config.size_array = int(np.sqrt(n_traps))

    '''
	Here I calculate the atom losses from one image to another
	'''
    atom_loss_bool = assembled_bool - initial_bool
    # print('assembled_bool - Initial', atom_loss_bool)

    '''
    x_values: scanned parameter, dimensions are [run nb]
	initial_bool: initially filled traps, dimensions are [run nb, traps ndx]
	assembled_bool: finally occupied traps after all rearrangements, dim: [run nb, traps ndx]
	assembled_list_bool: all occupation after rearrangement iterations, dim: [rearrang. iteration, run nb, traps ndx]
	recapture_bool: recaptured atoms after quantum manipulation, dim: [run nb, traps ndx]
	'''

    print('\nInitial bool \n', initial_bool)
    print('\nDone...!!!\n')
    return x_values, initial_bool, assembled_bool, assembled_list_bool, recapture_bool, atom_loss_bool
