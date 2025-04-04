import networkx as nx

def build_visual_graph(image_files, neighbors, distances, connect_temporal=False):
    """
    image_files: list of image identifiers (used as node labels)
    neighbors: (N, k) top-k neighbor indices
    distances: (N, k) L2 distances
    connect_temporal: whether to add edges between consecutive frames in same path
    """
    G = nx.Graph()
    N = len(image_files)

    # Add all images as nodes
    for i, fname in enumerate(image_files):
        G.add_node(i, filename=fname)

    # Add edges from FAISS nearest neighbors
    for i in range(N):
        for j in range(neighbors.shape[1]):
            ni = neighbors[i, j]
            dist = distances[i, j]
            G.add_edge(i, ni, weight=dist)

    #  Add temporal edges within the same path
    if connect_temporal:
        for i in range(1, N):
            cur_path = image_files[i].split("/")[0]
            prev_path = image_files[i - 1].split("/")[0]
            if cur_path == prev_path:
                G.add_edge(i, i - 1, weight=0.5)  # temporal edge with low weight

    return G

def find_shortest_path(G, start_idx, goal_idx):
    try:
        path = nx.astar_path(G, source=start_idx, target=goal_idx, weight='weight')
        return path
    except nx.NetworkXNoPath:
        print("No path found.")
        return None

