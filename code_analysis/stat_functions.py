import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def get_excitation_proba(folder):
    """
    x_values correspond to the scanned parameter, for instance the waiting time for the lifetime.
    """

    x_values = np.load(folder+'/x_values.npy')
    # assembled_bools_0 = np.load(folder+'/assembled_bool_0.npy')
    assembled_bools = np.load(folder+'/assembled_bool.npy')
    final_bools = np.load(folder+'/recap_bools.npy')
    selected_rois = np.loadtxt(folder+'/selectedROI.dat')
    selected_rois = selected_rois.astype(int)
    # selected_rois = np.arange(0,54,1)
    x_values_unique = np.unique(x_values)
    # x_values_unique = np.arange(1140,1148,1)

    N_atoms = len(selected_rois)  # len(assembled_bools[0])
    single_excitation_list = []
    N_list = []

    for x in x_values_unique:
        N_list_temp = []
        single_excitation_list_temp = []
        current_run_list = np.argwhere(x_values == x)

        for atom in range(N_atoms):
            atom_ini = assembled_bools[current_run_list, selected_rois[atom]]
            atom_fin = final_bools[current_run_list, selected_rois[atom]]
            N_run = 0
            single_excitation = 0

            for i in range(len(atom_ini)):
                N_run += atom_ini[i][0]
                single_excitation += atom_fin[i][0]

            if N_run != 0:
                single_excitation = 1-float(single_excitation)/float(N_run)
            else:
                single_excitation = np.nan

            N_list_temp.append(N_run)
            single_excitation_list_temp.append(single_excitation)

        N_list.append(N_list_temp)

        single_excitation_list.append(single_excitation_list_temp)

    N_list = np.asarray(N_list)
    single_excitation_list = np.asarray(single_excitation_list)
    # label_list = selected_rois+1

    return x_values_unique, single_excitation_list, N_list


def get_init_filling(initial_bools):
    """Returns the initial filling of the array.

    Args:
        initial_bools (np.array): an array with all the initial atom presences

    Returns:
        total_fill_R (float): total filling factor for the entire array
        trap_fill_R (np.array): filling factor for each single trap
    """
    n_run = initial_bools.shape[0]  # number of runs
    n_trap = initial_bools.shape[1]  # number of traps
    R_per_trap = np.sum(initial_bools, 0) / n_run
    R_total = np.sum(initial_bools) / n_run / n_trap
    return R_per_trap, R_total


def get_init_filling_rel(initial_bools):
    """Returns the initial filling of the array.

    Args:
        initial_bools (np.array): an array with all the initial atom presences

    Returns:
        total_fill_R (float): total filling factor for the entire array
        trap_fill_R (np.array): filling factor for each single trap
    """
    n_run = initial_bools.shape[0]  # number of runs
    n_trap = initial_bools.shape[1]  # number of traps
    R_per_trap = (np.sum(initial_bools, 0) / n_run)-0.5
    R_total = np.sum(initial_bools) / n_run / n_trap - 0.5
    return R_per_trap, R_total


def get_assemble_filling(assembled_bools, selected_rois):
    """Returns the filling of the array once rearranged.

    Args:
        assembled_bools (np.array): an array with all the rearranged atom presences

    Returns:
        total_fill_R (float): total filling factor for the target traps
        trap_fill_R (np.array): filling factor for each single trap
    """
    n_run = assembled_bools.shape[0]  # number of runs
    n_trap = assembled_bools.shape[1]  # number of traps
    R_per_trap = np.sum(assembled_bools, 0) / n_run
    R_total = np.sum(R_per_trap[selected_rois]) / len(selected_rois)
    off_roi = np.delete(R_per_trap, selected_rois)
    R_offroi = np.sum(off_roi) / (n_run - len(selected_rois))
    return R_per_trap, R_total, R_offroi


def get_assemble_filling_postselected(initial_bools, assembled_bools, selected_rois, n_defect_thresh):
    """Returns the filling of the array once rearranged.

    Args:
        assembled_bools (np.array): an array with all the rearranged atom presences

    Returns:
        total_fill_R (float): total filling factor for the target traps
        trap_fill_R (np.array): filling factor for each single trap
    """

    n_run = assembled_bools.shape[0]  # number of runs
    n_trap = assembled_bools.shape[1]  # number of traps

    # Exclude run with no rearrangement (not enough atoms)
    success_run_list = []
    for k in range(n_run):
        if np.sum(np.abs(initial_bools[k, :] - assembled_bools[k, :])) >= n_defect_thresh:
            success_run_list.append(k)

    R_per_trap = np.sum(
        assembled_bools[success_run_list, :], 0) / len(success_run_list)
    R_total = np.sum(R_per_trap[selected_rois]) / len(selected_rois)
    off_roi = np.delete(R_per_trap, selected_rois)
    R_offroi = np.sum(off_roi) / (len(success_run_list) - len(selected_rois))
    assembled_bools_ps = assembled_bools[success_run_list, :]
    return R_per_trap, R_total, R_offroi, assembled_bools_ps


def get_assemble_filling_rel(assembled_bools, selected_rois):
    """Returns the filling of the array once rearranged.

    Args:
        assembled_bools (np.array): an array with all the rearranged atom presences

    Returns:
        total_fill_R (float): total filling factor for the target traps
        trap_fill_R (np.array): filling factor for each single trap
    """
    n_run = assembled_bools.shape[0]  # number of runs
    n_trap = assembled_bools.shape[1]  # number of traps
    R_per_trap = np.sum(assembled_bools, 0) / n_run - 0.5
    R_total = np.sum(R_per_trap[selected_rois]) / len(selected_rois) - 0.5
    off_roi = np.delete(R_per_trap, selected_rois)
    R_offroi = np.sum(off_roi) / (n_run - len(selected_rois)) - 0.5
    return R_per_trap, R_total, R_offroi


def compute_failed_transfer_proba(assembled_bool, target_trap_list):
    """Compute the probability of failing to bring an atom to the target position
    for each starting trap.

    We assume that the atoms are dumped!

    We call 'moves', whenever an atom is moved from one trap to another. We call
    'dumping' when an atom is moved from trap to a dumpsite (and lost).

    Args:
        assembled_bool (np.array): an array with all the rearranged atom presences,
                                      with an optional post selection
        target_trap_list (np.array): list of traps to be filled by the rearranger

    Returns: 
        failed_transfer_from (np.array): proba of failure for transfer starting from
                                         each trap
        fail_move_proba (float): the total probability of failing a move
        moves_per_traps (np.array): The average number of move for each of the traps
        failed_dump_from (np.array): proba of failure for dumping an atom from
                                         each of the traps
        fail_dump_proba (float): the total probability of failing a dumping
        dumps_per_traps (np.array): The average number of dumping for each of the traps
    """
    n_run = assembled_bool.shape[0]
    failed_transfer_from = np.zeros(trap_nb)
    failed_dump_from = np.zeros(trap_nb)
    moves_per_traps = np.zeros(trap_nb)
    dumps_per_traps = np.zeros(trap_nb)
    total_move_nb = 0
    total_dump_nb = 0
    total_move_failed = 0
    total_dump_failed = 0

    for i in range(n_run):
        for move in moves[i]:
            start_site = move[0]  # trap index
            end_site = move[-1]  # trap index
            # in case it failed to transfer to the target trap
            if end_site < trap_nb:
                total_move_nb += 1
                moves_per_traps[start_site] += 1
                if int(assembled_bool[i][end_site]) == 0:
                    failed_transfer_from[start_site] += 1
                    total_move_failed += 1
            # in case it fails to take the atom to the dumpsite
            else:
                total_dump_nb += 1
                dumps_per_traps[start_site] += 1
                if int(assembled_bool[i][start_site]) == 1:
                    failed_dump_from[start_site] += 1
                    total_dump_failed += 1

    failed_transfer_from /= n_run
    failed_dump_from /= n_run
    moves_per_traps /= n_run
    dumps_per_traps /= n_run
    fail_move_proba = total_move_failed / total_move_nb
    fail_dump_proba = total_dump_failed / total_dump_nb

    return failed_transfer_from, fail_move_proba, moves_per_traps,           failed_dump_from, dumps_per_traps, fail_dump_proba


def plot_heatmap(data, total_proba, limits=None, main_res_str='Total probability', is_disable_cell=None):
    """
    Create a heatmap from a numpy array.

    Args:
        data (np.array): A 2D numpy array of shape (N, M).
        total_proba (float): total probability. Only used for title
        limits ((float,float)): lower and upper limit of the color scale
    """
    plt.figure(figsize=(10, 10))
    # create the Axes instance
    ax = plt.gca()

    # Plot the heatmap
    kw = {}
    if limits is not None:
        kw['vmin'] = limits[0]
        kw['vmax'] = limits[1]

    im = ax.imshow(data, cmap="afmhot", **kw)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Probability [-]", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[1]))
    # ... and label them with the respective list entries.
    row_labels = [str(i) for i in reversed(range(data.shape[0]))]
    col_labels = [str(i) for i in range(data.shape[1])]
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("{res} = {x:.4f}".format(res=main_res_str, x=total_proba))

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.

    threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    # kw = dict(horizontalalignment="center",
    #           verticalalignment="center")
    kw = dict(ha="center",
              va="center")

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    textcolors = ["black", "white"]
    valfmt = matplotlib.ticker.StrMethodFormatter("{x:.3f}")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            if is_disable_cell is None:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            else:
                if is_disable_cell[i][j]:
                    text = im.axes.text(j, i, 'NA', **kw)
                else:
                    text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
