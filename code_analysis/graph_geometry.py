"""
Date: May 14, 2020

Version = 2

Author: KNS

Here, we save the utilities to do the graph
"""

from scipy.spatial import Delaunay, Voronoi
import networkx as nx
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


def find_closest_Vor_points(trap_list, vor):
    """given a trap list and a voronoi diagram, return the closest voronoi node to each trap list"""
    distances = []
    index_min = []

    for j in range(len(trap_list)):
        distances = []
        for i in vor.vertices:

            distance = ((i[0]-trap_list[j][0])**2 +
                        (i[1]-trap_list[j][1])**2)**0.5
            distances.append(distance)
        index_min.append(np.argmin(distances))

    return vor.vertices[index_min]


def find_Dumpsites(trap_list):
    vor = Voronoi(trap_list)
    dumpsites = find_closest_Vor_points(trap_list, vor)
    return dumpsites


def find_slalomCoordinates(coordinates, trap_list):

    vor = Voronoi(trap_list)

    index_min_hor = np.where(vor.vertices[:, 0] < np.min(trap_list[:, 0]))
    index_max_hor = np.where(vor.vertices[:, 0] > np.max(trap_list[:, 0]))
    index_min_ver = np.where(vor.vertices[:, 1] < np.min(trap_list[:, 1]))
    index_max_ver = np.where(vor.vertices[:, 1] > np.max(trap_list[:, 1]))

    index_all = np.concatenate(
        (index_min_hor, index_max_hor, index_min_ver, index_max_ver), axis=-1)
    vor_clean = np.delete(vor.vertices, index_all, 0)

    x, y = zip(*sorted(zip(trap_list[:, 1], coordinates[:, 0])))
    f_hor = interp1d(x, y)
    x, y = zip(*sorted(zip(trap_list[:, 0], coordinates[:, 1])))
    f_ver = interp1d(x, y)

    vor_coord = []
    for i in vor.vertices:
        if i in vor_clean:
            vor_coord.append([f_hor(i[1]), f_ver(i[0])])
        else:
            # anyways this coordinate is never used since it is not in the graph! we cannot remove it however, s
            vor_coord.append([0, 0])
    vor_coord = np.array(vor_coord)  # since some things rely on the indices...

    pos = np.concatenate((coordinates, vor_coord))

    return pos


def compute_Slalomgraph(trap_list):
    #	"""
    #	This function should return the graph that is the Voronoi graph + single connections to trap coordinates locations. Then on this we can do "slalom" moves
    #
    #	"""
    vor = Voronoi(trap_list)
    return 0


def return_flat_edge(tria, traplist, min_dist):
    """
    This function is supposed to give you edges, that don't respect a minimal passing distance, because they are too close to another atom
    This is surely supotimally coded, but we don't care. I look at what is the longest side in each triangle, then I check the height of the
    triangle with respect to this side using the crossproduct, and is the height is below min_dist, I return it.

    """
    tooflatedges = []
    longedge = []
    for i in tria.simplices:
        triangle = traplist[i]

        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]

        sides = []
        a3 = np.linalg.norm(p2-p3)
        sides.append(a3)
        a2 = np.linalg.norm(p3-p1)
        sides.append(a2)
        a1 = np.linalg.norm(p2-p1)
        sides.append(a1)

        if max(sides) == a1:
            longedge.append([i[0], i[1]])
            d1 = np.abs(np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1))
            if d1 < min_dist:
                tooflatedges.append([i[0], i[1]])

        if max(sides) == a2:
            longedge.append([i[0], i[2]])
            d2 = np.abs(np.cross(p2-p1, p3-p1)/np.linalg.norm(p3-p1))
            if d2 < min_dist:
                tooflatedges.append([i[0], i[2]])

        if max(sides) == a3:
            longedge.append([i[1], i[2]])
            d3 = np.abs(np.cross(p2-p1, p3-p2)/np.linalg.norm(p3-p2))
            if d3 < min_dist:
                tooflatedges.append([i[1], i[2]])

    return tooflatedges


def remove_flat_edges(triangulation, trap_list, min_dist, graph):

    edgetoremove = return_flat_edge(triangulation, trap_list, min_dist)
    for edge in edgetoremove:
        graph.remove_edge(edge[0], edge[1])

    print("removed ", len(edgetoremove),
          " edges, because of minimal passing distance d= ", min_dist)


def compute_slalom_triangulation(trap_list):
    """
    Idea behind this: Compute the Voronoi graph. Have an interconnected Voronoi graph, add from each trap position one edge to the closest Voronoi node.
    Then, we can move along the Voronoi graph (slalom in between the atoms), but won't go "through" another atom, since there is only one edge from each
    trap position
    """

    vor = Voronoi(trap_list)

    distances = []
    index_min = []

    index_min_hor = np.where(vor.vertices[:, 0] < np.min(trap_list[:, 0]))
    index_max_hor = np.where(vor.vertices[:, 0] > np.max(trap_list[:, 0]))
    index_min_ver = np.where(vor.vertices[:, 1] < np.min(trap_list[:, 1]))
    index_max_ver = np.where(vor.vertices[:, 1] > np.max(trap_list[:, 1]))

    index_all = np.concatenate(
        (index_min_hor, index_max_hor, index_min_ver, index_max_ver), axis=-1)
#    print(np.sort(index_all[0]))
    vor_clean = np.delete(vor.vertices, index_all, 0)

    for j in range(len(trap_list)):
        distances = []
        for i in vor.vertices:

            distance = ((i[0]-trap_list[j][0])**2 +
                        (i[1]-trap_list[j][1])**2)**0.5
            distances.append(distance)
        index_min.append(np.argmin(distances))

    graph_slalom = nx.Graph()
    for i in range(len(trap_list)):
        graph_slalom.add_edge(i, len(trap_list)+index_min[i], weight=(
            (vor.vertices[index_min[i]][0]-trap_list[i][0])**2+(vor.vertices[index_min[i]][1]-trap_list[i][1])**2)**0.5)
    # create a set for edges that are indexes of the points
    edges2 = set()
    # for Voronoi Vertice
    offset = len(trap_list)
    for i in vor.ridge_vertices:
        # sometimes the indices are negative, which is indicated by the dotted lines on the graph.
        if i[0] >= 0 and i[1] >= 0:
            # These then are not triangles and "open boarders". We don't want to take them into account
            # or move along them!
            #            if i[0] not in index_all[0] and i[1] not in index_all[0]:

            i[0] += offset
            i[1] += offset
            edge2 = sorted(i)
            edges2.add((edge2[0], edge2[1]))

    graph_slalom.add_edges_from(list(edges2))

    graph_slalom.remove_nodes_from(np.sort(index_all[0]+offset))

    traps_positions = np.concatenate((trap_list, vor_clean))
    # we need consecutive numbering, however, if we were to remove say the node 2, the numbering would be 0,1,3,4 which would be bad.
    # Therefore we create a mapping and relabel the nodes!
    mapping = dict(zip(sorted(graph_slalom.nodes()),
                       range(0, len(traps_positions))))

 #   print(mapping)

    graph_slalom_relabeled = nx.relabel_nodes(graph_slalom, mapping)


#    print("trap pos",len(traps_positions))
#    print("graph",len(graph_slalom.nodes()))
#    print("graph relabel",len(graph_slalom_relabeled.nodes()))
#    for i in np.sort(graph_slalom.nodes()):
#        print(i)
#	    nx.draw(graph_slalom_relabeled,pos=traps_positions)

    plot_graph(graph_slalom_relabeled, traps_positions)

    return graph_slalom_relabeled


def compute_triangulation(trap_list):
    '''This function computes the Delaunay triangulation, given the trap coordinates.
    '''
    # Get a delaunay triangulation and a Delaunay graph-------
    tri = Delaunay(trap_list)

    # create a set for edges that are indexes of the points
    edges = set()
    # for each Delaunay triangle
    for n in range(tri.nsimplex):
        # for each edge of the triangle
        # sort the vertices
        # (sorting avoids duplicated edges being added to the set)
        # and add to the edges set
        edge = sorted([tri.vertices[n, 0], tri.vertices[n, 1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.vertices[n, 0], tri.vertices[n, 2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.vertices[n, 1], tri.vertices[n, 2]])
        edges.add((edge[0], edge[1]))

    # --------------------------------------------------------------------------
    # make the graph
    graph_del = nx.Graph(list(edges))

    # weight the edges with the distance
    for i in graph_del.edges():
        graph_del[i[0]][i[1]]['weight'] = np.sum(
            (trap_list[i[0]]-trap_list[i[1]])**2)**0.5

    min_dist = 3
    remove_flat_edges(tri, trap_list, min_dist, graph_del)

    return graph_del


def plot_graph(graph, trap_list):

    plt.figure(figsize=(15, 15))
    pointIDXY = dict(zip(range(len(trap_list)), trap_list))
    nx.draw_networkx(graph, pointIDXY, with_labels=True, font_weight="bold")
    plt.show()


def compute_triangulation_periodic_n(traps_positions, nb_lattice_constant):
    '''This function computes the Delaunay triangulation, given the trap coordinates.
    '''

    distance_matrix = []
    for position in traps_positions:
        distance_line = np.sqrt(
            (position[0] - traps_positions[:, 0])**2 + (position[1] - traps_positions[:, 1])**2)
        distance_matrix.append(distance_line)

    distance_matrix = np.asarray(distance_matrix)
    distance_list = np.round(distance_matrix.flatten(), decimals=4)

    unique = np.unique(distance_list)
    spacing_list = unique[1:1+nb_lattice_constant]

    tri = Delaunay(traps_positions)
    edges = set()
    for vertex in tri.vertices:

        dist_1 = ((traps_positions[vertex[1], 0] - traps_positions[vertex[0], 0])**2 + (
            traps_positions[vertex[1], 1] - traps_positions[vertex[0], 1])**2)**0.5
        dist_2 = ((traps_positions[vertex[2], 0] - traps_positions[vertex[0], 0])**2 + (
            traps_positions[vertex[2], 1] - traps_positions[vertex[0], 1])**2)**0.5
        dist_3 = ((traps_positions[vertex[2], 0] - traps_positions[vertex[1], 0])**2 + (
            traps_positions[vertex[2], 1] - traps_positions[vertex[1], 1])**2)**0.5

        if np.any(spacing_list - np.round(dist_1, decimals=4) == 0.0):
            edge = sorted([vertex[0], vertex[1]])
            edges.add((edge[0], edge[1]))
        if np.any(spacing_list - np.round(dist_2, decimals=4) == 0.0):
            edge = sorted([vertex[0], vertex[2]])
            edges.add((edge[0], edge[1]))
        if np.any(spacing_list - np.round(dist_3, decimals=4) == 0.0):
            edge = sorted([vertex[1], vertex[2]])
            edges.add((edge[0], edge[1]))

    graph = nx.Graph(list(edges))

    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = np.sum(
            (traps_positions[edge[0]]-traps_positions[edge[1]])**2)**0.5

    return graph


def compute_triangulation_periodic(traps_positions):
    '''This function computes the Delaunay triangulation, given the trap coordinates.
    '''
    tri = Delaunay(traps_positions)
    spacing = ((traps_positions[1, 0] - traps_positions[0, 0]) **
               2 + (traps_positions[1, 1] - traps_positions[0, 1])**2)**0.5
    edges = set()
    for vertex in tri.vertices:

        dist_1 = ((traps_positions[vertex[1], 0] - traps_positions[vertex[0], 0])**2 + (
            traps_positions[vertex[1], 1] - traps_positions[vertex[0], 1])**2)**0.5
        dist_2 = ((traps_positions[vertex[2], 0] - traps_positions[vertex[0], 0])**2 + (
            traps_positions[vertex[2], 1] - traps_positions[vertex[0], 1])**2)**0.5
        dist_3 = ((traps_positions[vertex[2], 0] - traps_positions[vertex[1], 0])**2 + (
            traps_positions[vertex[2], 1] - traps_positions[vertex[1], 1])**2)**0.5

        if np.around(dist_1-spacing, decimals=4) == 0.0:
            edge = sorted([vertex[0], vertex[1]])
            edges.add((edge[0], edge[1]))
        if np.around(dist_2-spacing, decimals=4) == 0.0:
            edge = sorted([vertex[0], vertex[2]])
            edges.add((edge[0], edge[1]))
        if np.around(dist_3-spacing, decimals=4) == 0.0:
            edge = sorted([vertex[1], vertex[2]])
            edges.add((edge[0], edge[1]))

    graph = nx.Graph(list(edges))

    for edge in graph.edges():
        graph[edge[0]][edge[1]]['weight'] = np.sum(
            (traps_positions[edge[0]]-traps_positions[edge[1]])**2)**0.5

#    fig = plt.figure()
#    plt.title("triangulation")
#    pointsidxy = dict(zip(range(len(traps_positions)),traps_positions))
#    nx.draw_networkx(graph,pointsidxy,with_labels=True)
#    plt.show()

    return graph


def initialize_allpair_slalom(graph, trap_list):

    all_pairs_distances = dict(
        nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    all_pairs_paths = dict(nx.all_pairs_dijkstra_path(graph, weight='weight'))

    list_distances = []

    for i in range(len(all_pairs_distances.keys())):
        row = []
        for j in range(len(all_pairs_distances.keys())):
            row.append(all_pairs_distances[i][j])
        list_distances.append(row)
    list_distances = np.array(list_distances)

    #cut_list = list_distances[:len(trap_list),:len(trap_list)]

    return list_distances, all_pairs_paths


def initialize_allpair(graph):

    all_pairs_distances = dict(
        nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    all_pairs_paths = dict(nx.all_pairs_dijkstra_path(graph, weight='weight'))

    list_distances = []

    for i in range(len(all_pairs_distances.keys())):
        row = []
        for j in range(len(all_pairs_distances.keys())):
            row.append(all_pairs_distances[i][j])
        list_distances.append(row)
    list_distances = np.array(list_distances)

    return list_distances, all_pairs_paths
