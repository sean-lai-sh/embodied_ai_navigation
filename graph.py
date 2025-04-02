import networkx as nx
import numpy as np

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


import networkx as nx

def build_visual_graph_with_actions(steps, connect_visual=False, visual_neighbors=None, visual_distances=None):
    G = nx.DiGraph()
    
    for i in range(len(steps) - 1):
        curr_img = steps[i]["image"]
        next_img = steps[i + 1]["image"]
        action = steps[i + 1]["action"][0]  # assumes one action, but extendable

        if action != "IDLE":  # skip idle transitions
            G.add_edge(curr_img, next_img, action=action, weight=1)

    if connect_visual and visual_neighbors is not None:
        # Optional: add visual similarity-based edges
        for i, neighbors in enumerate(visual_neighbors):
            for j, neighbor in enumerate(neighbors):
                dist = visual_distances[i][j]
                src = steps[i]["image"]
                tgt = steps[neighbor]["image"]
                if not G.has_edge(src, tgt):  # only add if not already temporally connected
                    G.add_edge(src, tgt, action="VISUAL", weight=dist)

    return G


def netvlad_distance_heuristic(u, v, netvlad_lookup):
    """
    A simple heuristic if you only have NetVLAD embeddings, not real-world coordinates.
    netvlad_lookup: dict or array that lets you get the descriptor for any image ID.
    We return the L2 distance between descriptors for nodes u and v.
    
    NOTE: This might not be strictly 'admissible' for true path cost. 
    If you need guaranteed optimal A*, you need a consistent or lower-bound heuristic.
    """
    desc_u = netvlad_lookup[u]
    desc_v = netvlad_lookup[v]
    return np.linalg.norm(desc_u - desc_v)


import networkx as nx

def run_a_star(graph, start_id, goal_id, netvlad_lookup=None):
    """
    Run A* on the graph from start_id to goal_id, optionally using a netvlad distance heuristic.
    If netvlad_lookup is None, we do A* with no heuristic -> effectively Dijkstra's search.
    """
    if netvlad_lookup is not None:
        # Define a function that NetworkX can call:
        def heuristic(u, v):
            return netvlad_distance_heuristic(u, v, netvlad_lookup)
        
        path = nx.astar_path(
            graph,
            start_id,  # source
            goal_id,   # target
            heuristic=heuristic,
            weight='weight'  # Use the 'weight' field on edges
        )
    else:
        # No heuristic, fallback to Dijkstra:
        path = nx.shortest_path(
            graph,
            start_id,
            goal_id,
            weight='weight'
        )
    return path
