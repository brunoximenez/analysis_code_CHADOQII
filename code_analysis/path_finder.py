"""
Created Wednesday May 14,2020

 Version = 13

author: KNS

This version incorporates the enhanced assembler algorithms.

# TO DO:

 -  in findMoves add something to return remaining_atoms as a list of trap
 numbers which are remaining unused
 -  add a dumpMoves function to dump these moves!

 - add other algorithms, e.g. the previous one.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def dumpMoves(remaining_atoms, dumpsites):
    dumpMoves = []
    for i in remaining_atoms:
        dumpMoves.append([i, dumpsites[i]])
    return dumpMoves


def dumpMoves_slalom(remaining_atoms, all_pairs_paths):
    dumpMoves = []

    for i in remaining_atoms:
        # luckily the list is ordered by length, so the first is the right dumpsite!
        dumpMoves.append([i, list(all_pairs_paths[i])[1]])
    return dumpMoves


def split_list(liste, intheway):
    indice = liste.index(intheway)
    slicea = liste[:indice+1]
    sliceb = liste[indice:]
    return [sliceb]+[slicea]


def findMoves_Hung_network(list_index_traps_loaded, list_index_trap_to_load, trap_list, all_pair_distances, alpha):
    '''
    This function is finding the moves from initial traps to target traps by
    using the hungarian algorithm. First we compute a costmatrixtime which has
    the distances between trap i and j at position ij, and use this as an input
    of the linear sum assignment. This results in a list of moves,
    the order is the same as the order of the target traps in trap_list
    '''

    # according to Daniel the costmatrix computation can be sped up.
    # double check if this 0.1 is really necessary
    costmatrix = []
    for i in list_index_trap_to_load:
        costmatrix.append(
            0.1*(all_pair_distances[list_index_traps_loaded, i])**alpha)
    costmatrix = np.array(costmatrix).astype(dtype=np.float16)

    row_ind, col_ind = linear_sum_assignment(costmatrix)

    moves_list = []
    for i in range(len(row_ind)):
        if list_index_traps_loaded[col_ind[i]] != list_index_trap_to_load[row_ind[i]]:
            moves_list.append([list_index_traps_loaded[col_ind[i]],
                               list_index_trap_to_load[row_ind[i]]])

    matched_atoms = list_index_traps_loaded[col_ind]
    temp = set(matched_atoms)
    remaining_atoms = [value for value in list_index_traps_loaded
                       if value not in temp]

    return moves_list, remaining_atoms


def findMoves_slalom_Hung_network(list_index_traps_loaded, list_index_trap_to_load, trap_list, all_pair_distances):
    '''
    This function is finding the moves from initial traps to target traps by using the hungarian algorithm. First we compute a costmatrixtime
    which has the distances between trap i and j at position ij, and use this as an input of the linear sum assignment. This results in a list of moves,
    the order is the same as the order of the target traps in trap_list
    '''
    atomsOut = np.setdiff1d(list_index_traps_loaded, list_index_trap_to_load)
    emptyTraps = np.setdiff1d(list_index_trap_to_load, list_index_traps_loaded)
    alpha = 1.0

    costmatrix = []
    for i in emptyTraps:
        costmatrix.append(0.1*(all_pair_distances[atomsOut, i])**alpha)
        #costmatrix.append(0.1*np.array([all_pair_distances[k][i] for k in list_index_traps_loaded])**alpha)
    costmatrix = np.array(costmatrix).astype(dtype=np.float16)

    row_ind, col_ind = linear_sum_assignment(costmatrix)

    moves_list = []
    for i in range(len(row_ind)):
        if atomsOut[col_ind[i]] != emptyTraps[row_ind[i]]:
            moves_list.append([atomsOut[col_ind[i]], emptyTraps[row_ind[i]]])
        # matched_atoms.append(list_index_traps_loaded[col_ind[i]])

    matched_atoms = atomsOut[col_ind]
    temp = set(matched_atoms)
    remaining_atoms = [value for value in atomsOut if value not in temp]

    # trap_list[list_index_trap_to_load[row_ind[i]]][0],trap_list[list_index_traps_loaded[col_ind[i]]][0]]
    return moves_list, remaining_atoms


def moves_along_graph_weighted_precalc(moves_list, all_pairs_paths):
    """
    This function gets the moves_list [initial trap number, final trap number]
    and returns the path in between. It does so using the precalculated paths
    all_pair_paths.
    """
    moves_list_delaunay = []
    for i in moves_list:    # shortest_path
        moves_list_delaunay.append(all_pairs_paths[i[0]][i[1]])

    return moves_list_delaunay


def postProcessHung(list_index_traps_loaded, moves_list_delaunay):
    '''
    This function reorders the collision-free assignment in a way that the
    moves are actually done in the right order. The problem is that the
    hungarian used before returns everything in a very unuseful order, by trap
    numbering. However, in our case the ordering is essential.

    The algorithm is surely not ideal but works as follows:
    You start with a list of moves that you get from the Hungarian, which is
    not particularily ordered.
    Take the first move from the list. If it passes the following conditions,
    put it in the new list of moves, if not, put it on the back of the old list.

    - If the end coordinate of the move you took is the first element of
    another move, it means an atom is at the target position.
      Hence, BAD move, put it at the back of the list
    - If the end coordinate is on the path of another move, we should not put
    it there, or else we will not be able to do the other move later on.
      Hence, BAD move, put it at the back of the list
    - If along the path of the move, there is a spot which is occupied, meaning
    it is the first element of another move in the list: BAD move

    '''
    #is_there_atom = np.copy(list_index_traps_loaded)
    bestorder = []
    moveslist_copy = list(moves_list_delaunay)
    moveslist = list(moves_list_delaunay)

    # print(moveslist)
    counter = 0
    while(len(moveslist_copy) > 0):
        for i in moveslist:
            bool_badelement = False
            # erstes element weg, wenn man es nicht benutzt hinten anhängen
            moveslist_copy.remove(i)

            for j in moveslist_copy:
                # if the end coordinate is in any other move it means it is either occupied or in the path of another atom, so it's a bad move
                if i[-1] in j:
                    bool_badelement = True
                    moveslist_copy.append(i)
                    break
                # if there is actually a spot that is occupied, it is a bad move
                if j[0] in i:
                    bool_badelement = True
                    moveslist_copy.append(i)
                    break
#            if bool_badelement:
#                moveslist_copy.append(i)
            else:
                bestorder.append(i)
                #is_there_atom = is_there_atom[is_there_atom != i[0]]
                #is_there_atom = np.append(is_there_atom,i[-1])
                # moveslist_copy.append(i)
        # print(counter)
        counter += 1

        if counter == 30:
            print("BUG: Endless Loop! ", counter)

            return bestorder, -1  # , is_there_atom
        moveslist = list(moveslist_copy)

    # print(counter)
    return bestorder, 0


def findMoves_shortestMove(list_index_traps_loaded, list_index_trap_to_load, trap_list):
    '''
    This is the Algorithm that is used currently on Chadoq since Barredo et al. 2016.
    '''

    atomsOut = []
    #print("traps_loaded ",traps_loaded)
    #print("list_index_trap_to_load ", list_index_trap_to_load)
    for i in list_index_traps_loaded:
        if i not in list_index_trap_to_load:
            atomsOut.append(i)
    atomsOut = np.array(atomsOut)
    #print("atomsOut2 ",atomsOut2)

    emptyTraps = []
    for i in list_index_trap_to_load:
        if i not in list_index_traps_loaded:
            emptyTraps.append(i)
    emptyTraps = np.array(emptyTraps)

    n_emptyTraps = len(emptyTraps)
    n_atomsOut = len(atomsOut)
    #print("number of empty traps: ",n_emptyTraps)
    #print("number of atoms outside: ",n_atomsOut)
    #print("number of traps ", len(trap_list))
    #print("number of occuopied traps ", list_index_traps_loaded)
    #print("traps to load ", list_index_trap_to_load)

    remaining_atoms = np.copy(atomsOut)  # list(atomsOut) #np.copy(atomsOut) #
#     print("number empty traps",n_emptyTraps)

    if n_emptyTraps > 0:
        # 1) matrix of distances
        distanceList = []
        for coord1 in trap_list[emptyTraps]:
            for coord2 in trap_list[atomsOut]:
                distanceList.append(
                    ((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2)**0.5)

        underlying_shape = (n_emptyTraps, n_atomsOut)

        sortingIndexes = np.argsort(distanceList)

        moves_list = []
        used_atomsOut = []
        filled_emptyTraps = []

        number_filled_traps = 0
        i = 0
        # Keep finding the good moves until you filled all the empty traps
        while number_filled_traps < n_emptyTraps:
            ind = sortingIndexes[i]

            # Get from this index the atom out and the empty trap
            empty_trap_index, atom_out_index = np.unravel_index(
                ind, underlying_shape)
            empty_trap = emptyTraps[empty_trap_index]
            atom_out = atomsOut[atom_out_index]

            # If we already used the atoms or filled this trap. continue
            if used_atomsOut.count(atom_out_index) != 0 or filled_emptyTraps.count(empty_trap_index) != 0:
                # Let's move to the next shortest distance
                i += 1
                continue

            # Pop the atom_out from the remaining atom list
            #print("atom out ", atom_out)
            #print("remaining atoms ",remaining_atoms)
            # remaining_atoms.remove(atom_out)
            remaining_atoms = remaining_atoms[remaining_atoms != atom_out]
            #print("remaining atoms ",remaining_atoms)

            # Add the atoms and the trap to the used and filled list
            used_atomsOut.append(atom_out_index)
            filled_emptyTraps.append(empty_trap_index)

            moves_list.append([atom_out, empty_trap])

            # We filled an empty trap, let's move to the next shortest distance
            number_filled_traps += 1
            i += 1
    else:
        moves_list = []
        remaining_atoms = atomsOut

    return moves_list, remaining_atoms


def postProcessMoves(list_index_traps_loaded, list_index_trap_to_load, moves_list_delaunay):
    '''
    The postprocess Function currently employed on Chadoq (from Barredo et al. 2016)
    '''

    modified = True
    loop = 0
    loop_max = 10

    # 1) Run the loop until there are no modifications
    while modified and loop < loop_max:
        loop += 1
        #print("loop ",loop)
        # Initialize to false, will change if there is a modification
        modified = False
        # make copy of arrays and list which will be updated when looping
        is_there_atom_2d_copy = np.copy(list_index_traps_loaded)

        new_moves_list = list(moves_list_delaunay)
        # print(is_there_atom_2d_copy)
        for move in moves_list_delaunay:
            #print("atom positions: ", is_there_atom_2d_copy)
            atoms_in_the_path_list = []

            for i in range(-2, -len(move), -1):

                if(move[i] in is_there_atom_2d_copy):
                    # print("hi")
                    atoms_in_the_path_list.append(move[i])

            number_atoms_in_the_path = len(atoms_in_the_path_list)
            #print("number_atoms in  the path ",number_atoms_in_the_path)

            if number_atoms_in_the_path == 0:
                # See Note A : updating the atoms positions
                is_there_atom_2d_copy = is_there_atom_2d_copy[is_there_atom_2d_copy != move[0]]
                # add the final position only if it is a trap
                if move[-1] in list_index_trap_to_load:

                    is_there_atom_2d_copy = np.append(
                        is_there_atom_2d_copy, move[-1])

                continue
            else:
                #modified = True
                #print("move ",move)
                #print("atoms in the path ",atoms_in_the_path_list,atoms_in_the_path_list[0])

                # a) I remove the bad moves from the list
                bad_moves_index = new_moves_list.index(move)
                part1 = new_moves_list[:bad_moves_index]
                part2 = new_moves_list[bad_moves_index+1:]

                good_moves_list = split_list(move, atoms_in_the_path_list[0])
                atoms_in_the_path_list.pop(0)
                #print("good moves list" ,good_moves_list)

                for j in range(len(atoms_in_the_path_list)):
                    # print("hi")
                    newmoves = []
                    for i in good_moves_list:
                        if (atoms_in_the_path_list[0] in i):
                            split1, split2 = split_list(
                                i, atoms_in_the_path_list[0])
                            newmoves.append(split1)
                            newmoves.append(split2)
                        else:
                            newmoves.append(i)
                    good_moves_list = list(newmoves)
                    atoms_in_the_path_list.pop(0)
                #print("good moves list" ,good_moves_list)
                for goodmove in good_moves_list:
                    is_there_atom_2d_copy = is_there_atom_2d_copy[is_there_atom_2d_copy != goodmove[0]]
                    if goodmove[-1] in list_index_trap_to_load:
                        is_there_atom_2d_copy = np.append(
                            is_there_atom_2d_copy, goodmove[-1])

                new_moves_list = new_moves_list = part1 + good_moves_list + part2
        if loop == loop_max:
            print('BUUUUG too much loop')

    return new_moves_list  # ,is_there_atom_2d_copy


def review_post_process(list_path, version):
    new_path_list = []
#    list_sources = []
#    for path in list_path:
#        list_sources.append(path[0])
    list_path_copy = list(list_path)
    all_path_scanned = False
    index_path_scanned = 0
#    joker_iteration = 0
    while not all_path_scanned:

        path = list_path_copy[index_path_scanned]
        target = path[-1]
#        print(index_path_scanned)

#        following_path = list_path[index_path_scanned+1]

        future_sources = []
        future_targets = []

        for future_path in list_path_copy[index_path_scanned+1:]:
            future_sources.append(future_path[0])
            future_targets.append(future_path[-1])

        if target in future_sources:
            index_path_to_merge = index_path_scanned + \
                1 + future_sources.index(target)
            path_to_merge = list_path_copy[index_path_to_merge]

            traps_in_common = []
            for trap in path:
                if trap in path_to_merge:
                    traps_in_common.append(trap)

            connecting_point = traps_in_common[0]
            index_1 = path.index(connecting_point)
            index_2 = path_to_merge.index(connecting_point)
            new_path = path[:index_1] + path_to_merge[index_2:]


# version 1, first try to postpone, else try to do it now
##
####
            if version == 'postpone':
                good_to_postpone = True
                compteur = 0
                while good_to_postpone and compteur < index_1:
                    site = path[compteur]
                    if site in future_targets[:future_sources.index(target)]:
                        good_to_postpone = False
                    compteur += 1

                if good_to_postpone:
                    compteur = index_path_scanned+1
                    while good_to_postpone and compteur < index_path_to_merge:
                        future_path = list_path_copy[compteur]
                        if path[0] in future_path:
                            good_to_postpone = False
                        compteur += 1

                if good_to_postpone:
                    list_path_copy[index_path_to_merge] = new_path
                    list_path_copy.pop(index_path_scanned)
                else:
                    #                print("coucou")
                    #                new_path_list.append(path)
                    #                index_path_scanned += 1

                    good_to_do_now = True
                    compteur = index_2
                    while good_to_do_now and compteur < len(path_to_merge):
                        site = path_to_merge[compteur]
                        if site in future_sources[:future_sources.index(target)]:
                            good_to_do_now = False
                        compteur += 1

                    if good_to_do_now:
                        compteur = index_path_scanned+1
                        while good_to_do_now and compteur < index_path_to_merge:
                            future_path = list_path_copy[compteur]
                            if new_path[-1] in future_path:
                                good_to_do_now = False
                            compteur += 1

                    if good_to_do_now:
                        #                    print("good to do now")
                        list_path_copy[index_path_scanned] = new_path
                        list_path_copy.pop(index_path_to_merge)
                    else:
                        #                    print("coucou")
                        new_path_list.append(path)
                        index_path_scanned += 1
#


# version 2, first try to do it now, else try to postpone
##
####
            elif version == 'do_now':
                good_to_do_now = True
                compteur = index_2
                while good_to_do_now and compteur < len(path_to_merge):
                    site = path_to_merge[compteur]
                    if site in future_sources[:future_sources.index(target)]:
                        good_to_do_now = False
                    compteur += 1

                if good_to_do_now:
                    compteur = index_path_scanned+1
                    while good_to_do_now and compteur < index_path_to_merge:
                        future_path = list_path_copy[compteur]
                        if new_path[-1] in future_path:
                            good_to_do_now = False
                        compteur += 1

                if good_to_do_now:
                    #                    print("good to do now")
                    list_path_copy[index_path_scanned] = new_path
                    list_path_copy.pop(index_path_to_merge)

                else:
                    #                print("coucou")
                    #                new_path_list.append(path)
                    #                index_path_scanned += 1

                    good_to_postpone = True
                    compteur = 0
                    while good_to_postpone and compteur < index_1:
                        site = path[compteur]
                        if site in future_targets[:future_sources.index(target)]:
                            good_to_postpone = False
                        compteur += 1

                    if good_to_postpone:
                        compteur = index_path_scanned+1
                        while good_to_postpone and compteur < index_path_to_merge:
                            future_path = list_path_copy[compteur]
                            if path[0] in future_path:
                                good_to_postpone = False
                            compteur += 1

                    if good_to_postpone:
                        list_path_copy[index_path_to_merge] = new_path
                        list_path_copy.pop(index_path_scanned)

                    else:
                        #                        print("coucou")
                        new_path_list.append(path)
                        index_path_scanned += 1

        else:
            new_path_list.append(path)
            index_path_scanned += 1

# end strategy

        if index_path_scanned == len(list_path_copy)-1:
            all_path_scanned = True

    new_path_list.append(list_path_copy[-1])
#    print(joker_iteration)

    return new_path_list


def compute_moves_for_benchmarking(isThereAtomList, n_vert):
    """
    The benchmarking experiment: Have a square array, in each column, dump the
    first atom, then move all the atoms one down.
    """
    list_index_traps_loaded = []
    for i, isThereAtom in enumerate(isThereAtomList):
        if isThereAtom:
            list_index_traps_loaded.append(i)
    list_index_traps_loaded = np.array(list_index_traps_loaded)

    list_index_dumpsites = np.arange(
        len(isThereAtomList), 2*len(isThereAtomList))

    dumpPositions = []
    moves = []
    for position in list_index_traps_loaded:
        if position % n_vert == 0:
            dumpPositions.append(position)
        else:
            moves.append([position, position-1])
    moves_dump = dumpMoves(dumpPositions, list_index_dumpsites)

    totalMoves = moves_dump + moves

    return totalMoves


def compute_moves_for_benchmarking2(isThereAtomList, filledTrapList, nvert):
    """
    Second benchmarking experiment. Just have two rows. We don't want to remove any
    atoms, since if this is not succesful, it influences the assembly afterwards, which then is also not succesful.
    Therefore, for each pair, we just do a move, if there is an atom in one of the traps, if we have a doublon, we ignore it.
    We can decide if we want to do a row or just two trap, it should not matter.
    """
    list_index_traps_loaded = []
    for i, isThereAtom in enumerate(isThereAtomList):
        if isThereAtom:
            list_index_traps_loaded.append(i)
    list_index_traps_loaded = np.array(list_index_traps_loaded)

    list_index_trapsites = np.arange(len(isThereAtomList))

    moves = []
    pairs = []

    for i in range(0, len(list_index_trapsites), 2):
        pairs.append([i, i+1])

    for pair in pairs:

        if (pair[0] in list_index_traps_loaded) and (pair[1] not in list_index_traps_loaded):
            moves.append([pair[0], pair[1]])
        if pair[0] not in list_index_traps_loaded and pair[1] in list_index_traps_loaded:
            moves.append([pair[1], pair[0]])

    return moves


def order(isThereAtomList, filledTrapsList, trap_list, all_pairs_distances,
          all_pairs_paths, algorithm, dump_moves_bool):
    """
    This function will be the heart of the assembler.
        - isThereAtomList : list of booleans
            isThereAtomList[i] indicates if there is an atom in trap i
        - filledTrapsList : list of booleans
            filledTrapsList[i] indicates if we want an atom in trap i

    Maybe in the future: Add algorithm, then you could choose between different functions!

    Return :
            - totalMoves : a list of moves. Each moves_list can contain several moves
            A moves_list will move an atom to an empty trap
            totalMoves = [moves_list1, moves_list2, moves_list3]


    """

    number_atoms_init = np.sum(isThereAtomList)
    number_atoms_goal = np.sum(filledTrapsList)
#     print('number_atoms_init',number_atoms_init, 'number_atoms_goal', number_atoms_goal)
    if number_atoms_init < number_atoms_goal:
        print('Not enough atoms : gives up... : (')
        totalMoves = []
        return totalMoves

    # From the boolean lists, we would like to have just a list of trap numbers.
    # list_index_traps_loaded,list_index_trap_to_load,trap_list
    # still have to see how i would get the trap_list in here!

    # adding the dumping sites. ongoing numbering

    list_index_dumpsites = np.arange(len(trap_list), 2*len(trap_list))

    list_index_traps_loaded = np.where(np.array(isThereAtomList) == True)[
        0]  # i think this is twice as fast, let's try

    list_index_trap_to_load = np.where(np.array(filledTrapsList) == True)[0]

#     print('list_index_traps_loaded',list_index_traps_loaded)
#     print('list_index_trap_to_load',list_index_trap_to_load)

    if len(list_index_trap_to_load) > 0:
        if str(algorithm) == "LSAP alpha=2.0":
            alpha = 2.0

            moves_list, remaining_atoms = findMoves_Hung_network(
                list_index_traps_loaded, list_index_trap_to_load,
                trap_list, all_pairs_distances, alpha)
#             print('moves_list',moves_list)
#             print('remaining_atoms',remaining_atoms)

            # this calculates the path along the graph:
            # [initial trap #, trap # in between1,...,final trap #]
            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            # postprocessing to reorder the moves in a good always, so that
            # they are actually in a collisionless order
            # the algorithm should in principle not fail, but since it has this
            # while loop, I break after e.g. 30 loops and say it failed, or
            # anyways it takes too long
            postprocessedMoves, fail = postProcessHung(list_index_traps_loaded,
                                                       moves_list_alongPath)

            if fail != 0:
                print("hungarian postprocessing failed :(")
                return []

        elif algorithm == "Shortest-Move first":

            moves_list, remaining_atoms = findMoves_shortestMove(
                list_index_traps_loaded, list_index_trap_to_load, trap_list)

            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = postProcessMoves(
                list_index_traps_loaded, list_index_trap_to_load, moves_list_alongPath)

        elif algorithm == "SMF + Review postpr.":
            moves_list, remaining_atoms = findMoves_shortestMove(
                list_index_traps_loaded, list_index_trap_to_load, trap_list)

            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = postProcessMoves(
                list_index_traps_loaded, list_index_trap_to_load, moves_list_alongPath)

            if postprocessedMoves:
                if len(postprocessedMoves) > 1:
                    postprocessedMoves = review_post_process(
                        postprocessedMoves, "postpone")

        elif algorithm == "LSAP alpha=1.0":
            alpha = 1.0
            moves_list, remaining_atoms = findMoves_Hung_network(
                list_index_traps_loaded, list_index_trap_to_load, trap_list, all_pairs_distances, alpha)
           # print("remaining atoms",remaining_atoms)
            # it should be this!
            #moves_list,remaining_atoms = findMoves_Hung_network(list_index_traps_loaded,list_index_trap_to_load,trap_list,all_pairs_distances,alpha)

            # this calculates the path along the graph [initial trap number, trap number in between1,...,final trap number]
            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = postProcessMoves(
                list_index_traps_loaded, list_index_trap_to_load, moves_list_alongPath)

        elif algorithm == "LSAP a=1 + Review postpr.":
            alpha = 1.0
            moves_list, remaining_atoms = findMoves_Hung_network(
                list_index_traps_loaded, list_index_trap_to_load, trap_list, all_pairs_distances, alpha)
           # print("remaining atoms",remaining_atoms)
            # it should be this!
            #moves_list,remaining_atoms = findMoves_Hung_network(list_index_traps_loaded,list_index_trap_to_load,trap_list,all_pairs_distances,alpha)

            # this calculates the path along the graph [initial trap number, trap number in between1,...,final trap number]
            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = postProcessMoves(
                list_index_traps_loaded, list_index_trap_to_load, moves_list_alongPath)

            if postprocessedMoves:
                if len(postprocessedMoves) > 1:
                    postprocessedMoves = review_post_process(
                        postprocessedMoves, "postpone")

        elif algorithm == "Slalom":
            #            alpha = 1.0
            #            moves_list, remaining_atoms = findMoves_Hung_network(list_index_traps_loaded,list_index_trap_to_load,trap_list,all_pairs_distances,alpha)
            moves_list, remaining_atoms = findMoves_shortestMove(
                list_index_traps_loaded, list_index_trap_to_load, trap_list)
            # this calculates the path along the graph [initial trap number, trap number in between1,...,final trap number]
            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = moves_list_alongPath

        elif algorithm == "Slalom LSAP":
            moves_list, remaining_atoms = findMoves_slalom_Hung_network(
                list_index_traps_loaded, list_index_trap_to_load, trap_list, all_pairs_distances)
            moves_list_alongPath = moves_along_graph_weighted_precalc(
                moves_list, all_pairs_paths)

            postprocessedMoves = moves_list_alongPath

        else:
            print("Fail: Algorithm not yet implemented")
            postprocessedMoves = []
            remaining_atoms = []

        if dump_moves_bool:

            if algorithm == "Slalom" or algorithm == "Slalom LSAP":
                moves_dump = dumpMoves_slalom(remaining_atoms, all_pairs_paths)

            else:
                moves_dump = dumpMoves(remaining_atoms, list_index_dumpsites)
#            moves_dump = dumpMoves(remaining_atoms,list_index_dumpsites)
            totalMoves = postprocessedMoves+moves_dump
        else:
            totalMoves = postprocessedMoves

    else:
        #print("no atoms to load")
        if dump_moves_bool:

            if algorithm == "Slalom" or algorithm == "Slalom LSAP":
                totalMoves = dumpMoves_slalom(
                    list_index_traps_loaded, all_pairs_paths)
            else:
                totalMoves = dumpMoves(
                    list_index_traps_loaded, list_index_dumpsites)
#            totalMoves = dumpMoves(list_index_traps_loaded,list_index_dumpsites)
        else:
            totalMoves = []

    return totalMoves
