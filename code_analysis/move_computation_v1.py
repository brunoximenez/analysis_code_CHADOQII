import time
import numpy as np
import pandas as pd
import sys
import os
import networkx as nx
# print(sys.version)
import path_finder
import graph_geometry as graph_geo

data_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
print(data_path)
trap_file = os.path.join(data_path, 'trap_positions.dat')
sel_roi_file = os.path.join(data_path, 'selectedROI.dat')
trap_nb = None


def compute_moves(init_filled_traps, target_trap_list, trap_positions):
    """Compute the list of moves in a similar way as the Chadoq1 code.
    """
    run_nb = init_filled_traps.shape[0]

    # Get the lattice structure
    all_pairs_distances, all_pairs_paths, trap_positions =\
        generate_geometry(trap_positions)
    trap_nb = len(trap_positions)  # Take the dumpsites into account

    # Convert the input to the right format:
    init_fill_bool = np.full((run_nb, init_filled_traps.shape[1]), False)
    trap_only_bool = np.array(init_filled_traps, dtype=bool)

    # we need to add the dumpsites (empty)
    init_fill_bool = np.block([trap_only_bool, init_fill_bool])
    target_trap_bools = np.array([True if (i in target_trap_list) else False
                                  for i in range(trap_nb)])

    moves = []
    # Compute the actual moves
    for run_i in range(run_nb):
        #         print('init_fill_bool', init_fill_bool, init_fill_bool.shape)
        #         print('trap_only_bool', trap_only_bool, trap_only_bool.shape)
        #         print('init_fill_bool', init_fill_bool[run_i], init_fill_bool.shape)
        #         print('target_trap_bools', target_trap_bools, target_trap_bools.shape)
        #         print('processed init fill', np.where(np.array(init_fill_bool[run_i])==True)[0])
        #         print('all_pairs_distances',all_pairs_distances)
        #         print('all_pairs_paths', all_pairs_paths)
        #         print('trap_positions', trap_positions)
        i_move = path_finder.order(
            init_fill_bool[run_i],
            target_trap_bools,
            trap_positions,
            all_pairs_distances,
            all_pairs_paths,
            'LSAP alpha=2.0',
            True
        )
        moves.append(i_move)
    return moves


def import_traps():
    with open(trap_file) as f:
        ncols = len(f.readline().split(','))
    red = [i for i in range(1, ncols)]
    traps_pd = pd.read_csv(trap_file, header=0, delimiter=',', usecols=red)
    traps = pd.DataFrame.to_numpy(traps_pd)
    return traps[:, :2]


def import_sel_rois():
    sel_rois_pd = pd.read_csv(sel_roi_file, header=0, delimiter=',')
    sel_rois = pd.DataFrame.to_numpy(sel_rois_pd)
    return sel_rois.flatten()


def generate_geometry(trap_positions):

    all_coordinates = []
    all_pairs_distances = []
    all_pairs_paths = []
    mode = "Regular Array, #a=1"
    coordinates = trap_positions.copy()

    dumpsites = graph_geo.find_Dumpsites(coordinates)
    all_coordinates = np.concatenate((coordinates, dumpsites), axis=0)

    if mode == "Regular Array, #a=1":
        graph_del = graph_geo.compute_triangulation_periodic_n(
            trap_positions, 1)
    elif mode == 'Regular Array, #a=2':
        graph_del = graph_geo.compute_triangulation_periodic_n(
            trap_positions, 2)
    elif mode == "Slalom":
        graph_del = graph_geo.compute_slalom_triangulation(trap_positions)
    else:
        graph_del = graph_geo.compute_triangulation(trap_positions)

    num_components = nx.number_connected_components(graph_del)
    print("number of connected components: ", num_components)

    if(num_components == 1):
        all_pairs_distances, all_pairs_paths = graph_geo.initialize_allpair(
            graph_del)
    else:
        print('error while generating graph')

    return all_pairs_distances, all_pairs_paths, all_coordinates


if __name__ == '__main__':

    print(import_traps())
