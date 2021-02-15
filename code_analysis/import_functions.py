import numpy as np
import pandas as pd
import os
# ## imports functions

# In[23]:


def load(filename, header_return=False):
    '''
    Load the data in the file into a list
    '''

    data_list = []
    with open(filename, 'r') as f:
        # Get the header
        header = next(f)
        # Get the datas
        for line in f:
            # strip() removes the \n, split('\t') return a list whose elements where separated by a \t
            data_list.append(line.strip().split('\t'))
    f.close()

    data_list = np.asarray(data_list)

    if header_return:
        return data_list, header
    else:
        return data_list


def get_boolean(folder):

    x_values = np.load(folder+'/x_values.npy')
    initial_bools = np.load(folder+'/initial_bool.npy')
    assembled_bools = np.load(folder+'/assembled_bool.npy')
    final_bools = np.load(folder+'/recap_bools.npy')
    selected_rois = np.loadtxt(folder+'/selectedROI.dat')
    selected_rois = selected_rois.astype(int)

    return x_values, initial_bools, assembled_bools, final_bools, selected_rois

    return model-data


def parseFile(filename):

    with open(filename) as f:
        ncols = len(f.readline().split('\t'))
    # without reference_fluo
    red = [i for i in range(1, ncols)]
    data3 = pd.read_csv(filename, header=None, delimiter='\t', usecols=red)
    data = pd.DataFrame.to_numpy(data3)
    return data


def openROIfile():
    '''
    read and parse the ROI position from the file
    '''
    filename = 'ROI_positions.dat'
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

    data_list = np.asarray(data_list)

    data = np.asarray(data_list)

    number_of_rois = len(data)
    roi_positions = np.zeros((number_of_rois, 3))

    for index, line in enumerate(data):

        # Read the roi position
        pos = line[:6]
        pos = np.asarray(pos).astype('float').astype('int')

        roi_positions[index, 0] = pos[1]
        roi_positions[index, 1] = pos[2]
    return number_of_rois, roi_positions


def formatData(data, number_of_rois, threshold_global_fail, threshold_global_recap):
    '''
    threshold = data[:,1::5]
    initial_fluo = data[:,2::5]
    assembled_fluo = data[:,3::5]
    recap_fluo = data[:,4::5]
    reference_fluo = data[:,5::5] Not used

    1) Substract the threshold level
    from the initial, assembled and recapture fluorescence levels

    2) Binarize the data by checking if the fluorescence was above
    or below the threshold
    '''
    # 1)
    number_assemblies = int((len(data[0])-1)/number_of_rois - 4)
    #print("number assemblies",number_assemblies)
    repetition = int(4+number_assemblies)
    # print("repetition",repetition)

    # Use threshold with reference_fluo
    initial_fluo = data[:, 2::repetition] - \
        threshold_global_fail - data[:, 1::repetition]
    assembled_list_fluo = []
    for i in range(number_assemblies):
        assembled_fluo = data[:, 3+i::repetition] - \
            threshold_global_fail - data[:, 1::repetition]
        assembled_list_fluo.append(assembled_fluo)

    assembled_fluo = data[:, 2+number_assemblies::repetition] - \
        threshold_global_fail - data[:, 1::repetition]
    recap_fluo = data[:, 3+number_assemblies::repetition] - \
        threshold_global_recap - data[:, 1::repetition]

    N_atom = len(initial_fluo[0])
   # 2)
    assembled_list_bool = []
    # if use_threshold_check.isChecked():
    if False:
        initial_bool = initial_fluo > thresholds[:N_atom, 0]
        assembled_bool = assembled_fluo > thresholds[:N_atom, 0]
        recapture_bool = recap_fluo > thresholds[:N_atom, 1]
        for assembled in assembled_list_fluo:
            assembled_list_bool.append(assembled > thresholds[:N_atom, 0])

    else:
        initial_bool = initial_fluo > 0
        assembled_bool = assembled_fluo > 0
        recapture_bool = recap_fluo > 0
        for assembled in assembled_list_fluo:
            assembled_list_bool.append(assembled > 0)
    assembled_list_bool = np.array(assembled_list_bool)

    initial_bool = initial_bool.astype(int)
    assembled_bool = assembled_bool.astype(int)
    recapture_bool = recapture_bool.astype(int)
    assembled_list_bool = assembled_list_bool.astype(int)

    return initial_bool, assembled_bool, assembled_list_bool, recapture_bool
