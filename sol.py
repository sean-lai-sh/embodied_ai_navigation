# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import argparse

import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from natsort import natsorted

import logging

from FaiSS import FaissIndex
from graph import build_visual_graph
import networkx as nx
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "data/images/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")
        
        # Keyboard + FPV state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None

        # SIFT feature extractor
        self.sift = cv2.SIFT_create()
        
        # Data structures for NetVLAD
        self.sift_descriptors = None
        self.codebook = None
        self.database = None
        self.faiss_index = None
        self.graph = None
        # We'll store NetVLAD vectors by filename in a dict
        self.netvlad_lookup = None

        # Potentially load from disk if files exist
        # TODO: REMEMBER THAT POST TRAINING TO UN COMMENT THE FILES
        # if os.path.exists("sift_descriptors.npy"):
        #     self.sift_descriptors = np.load("sift_descriptors.npy")
        # if os.path.exists("codebook.pkl"):
        #     self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # if os.path.exists("database.pkl"):
        #     self.database = pickle.load(open("database.pkl", "rb"))
        # if os.path.exists("faiss_index.pkl"):
        #     self.faiss_index = pickle.load(open("faiss_index.pkl", "rb"))
        # if os.path.exists("graph.pkl"):
        #     self.graph = pickle.load(open("graph.pkl", "rb"))

        # Navigation
        self.goal = None

    def reset(self):
        """ Reset the player state and set up pygame. """
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """ Handle keyboard input for controlling the agent. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    # Unmapped key => show target images
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """ 2x2 grid display of the 4 'goal' images. """
        targets = self.get_target_images()
        if not targets or len(targets) < 4:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        color = (0, 0, 0)
        concat_img = cv2.line(concat_img, (h//2, 0), (h//2, w), color, 2)
        concat_img = cv2.line(concat_img, (0, w//2), (h, w//2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        cv2.putText(concat_img, 'Front View', (10, 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (h//2 + 10, 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (10, w//2 + 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (h//2 + 10, w//2 + 25), font, size, color, stroke, line)

        cv2.imshow('KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """ Called by the environment to set the 4 target images. """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """ Display an image from 'self.save_dir' given its numeric ID (string). """
        path = os.path.join(self.save_dir, f"{id}.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist at {path}")

    def compute_sift_features(self):
        """ Compute SIFT descriptors (aggregated) for all images in self.save_dir. """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        sift_descriptors = []
        for imgfile in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, imgfile))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)

    def run_a_star(self, source_id, target_id):
        """
        Run A* search on self.graph from source_id to target_id
        using a NetVLAD-based heuristic if self.netvlad_lookup is populated.
        Otherwise, fallback to nx.shortest_path (like Dijkstra).
        """
        if not hasattr(self, 'netvlad_lookup') or self.netvlad_lookup is None:
            # If no descriptor lookup is available, just use Dijkstra / BFS
            return nx.shortest_path(self.graph, source=source_id, target=target_id, weight='weight')

        def netvlad_heuristic(u, v):
            if u not in self.netvlad_lookup or v not in self.netvlad_lookup:
                return 0.0
            desc_u = self.netvlad_lookup[u]
            desc_v = self.netvlad_lookup[v]
            return np.linalg.norm(desc_u - desc_v)

        return nx.astar_path(
            self.graph,
            source=source_id,
            target=target_id,
            heuristic=netvlad_heuristic,
            weight='weight'
        )

    def get_netVLAD(self, img):
        """
        Compute a NetVLAD-style descriptor using SIFT descriptors + soft assignment to KMeans centroids.
        """
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.codebook.n_clusters * 128)

        centroids = self.codebook.cluster_centers_
        k = centroids.shape[0]
        d = des.shape[1]

        # Distances from each local descriptor to each cluster center
        dists = np.linalg.norm(des[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
        sigma = 1e-6 + np.std(dists)
        weights = np.exp(-dists**2 / (2 * sigma**2))  # (N, K)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        netvlad_feature = np.zeros((k, d), dtype=np.float32)
        for cluster_id in range(k):
            soft_weights = weights[:, cluster_id][:, None]  # shape (N, 1)
            residuals = des - centroids[cluster_id]
            weighted_residuals = soft_weights * residuals
            netvlad_feature[cluster_id] = weighted_residuals.sum(axis=0)

        # Flatten
        netvlad_feature = netvlad_feature.flatten()
        # Signed square root (power normalization)
        netvlad_feature = np.sign(netvlad_feature) * np.sqrt(np.abs(netvlad_feature))
        # L2 normalize
        netvlad_feature = netvlad_feature / (np.linalg.norm(netvlad_feature) + 1e-12)
        return netvlad_feature

    def netvlad_aggregation(descriptors, centroids):
        """
        A simpler (hard assignment) version of NetVLAD. Not used in this pipeline,
        but kept here for reference.
        """
        K, D = centroids.shape
        labels = np.argmin(np.linalg.norm(descriptors[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        vlad = np.zeros((K, D), dtype=np.float32)
        for i in range(K):
            if np.sum(labels == i) == 0:
                continue
            residuals = descriptors[labels == i] - centroids[i]
            vlad[i] = residuals.sum(axis=0)
        vlad = vlad.flatten()
        return vlad / np.linalg.norm(vlad)

    def get_neighbor(self, img, k=5):
        """
        For a given 'img', compute its NetVLAD descriptor, then query FAISS for the top-k neighbors.
        Return the first neighbor that actually exists in the graph's nodes.
        """
        desc = self.get_netVLAD(img)
        indices, _ = self.faiss_index.query(desc, k=k)
        for idx in indices:
            idx = str(idx)
            if not idx.endswith(".jpg"):
                idx += ".jpg"
            if self.graph.has_node(idx):
                return idx
        # Fallback: if none of the top-k neighbors are in the graph, return the first index
        return str(indices[0]) + ".jpg"

    def pre_nav_compute(self):
        """ Build or load the SIFT/NetVLAD/FAISS/Graph so we can run A* later. """
        if self.sift_descriptors is None:
            print("Computing SIFT features...")
            self.sift_descriptors = self.compute_sift_features()
            np.save("sift_descriptors.npy", self.sift_descriptors)
        else:
            print("Loaded SIFT features from sift_descriptors.npy")
        
        if self.codebook is None:
            print("Computing codebook (MiniBatchKMeans)...")
            self.codebook = MiniBatchKMeans(
                n_clusters=64,
                batch_size=5000,
                init='k-means++',
                n_init=5,
                verbose=1,
            ).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            print("Loaded codebook from codebook.pkl")

        # We'll use all .jpg files in self.save_dir
        exploration_observation = natsorted(
            [x for x in os.listdir(self.save_dir) if x.endswith('.jpg')]
        )

        # If we don't have a global descriptor database, build it
        if self.database is None:
            self.database = []
            print("Computing NetVLAD embeddings for each image...")
            for fname in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, fname))
                VLAD = self.get_netVLAD(img)
                self.database.append(VLAD)
            pickle.dump(self.database, open("database.pkl", "wb"))
        else:
            print("Loaded NetVLAD embeddings from database.pkl")

        # Build or load the FAISS index
        if self.faiss_index is None:
            print("Building FAISS index...")
            self.faiss_index = FaissIndex(dim=self.database[0].shape[0])
            self.faiss_index.build_index(self.database, image_ids=exploration_observation)
            pickle.dump(self.faiss_index, open("faiss_index.pkl", "wb"))
        else:
            print("Loaded FAISS index from faiss_index.pkl")

        # Build or load the k-NN graph
        if self.graph is None:
            print("Building k-NN graph from FAISS neighbors...")
            neighbors, distances = self.faiss_index.batch_query(self.database, k=10)
            # Use your function build_visual_graph(...) or build_visual_graph_with_actions(...)
            self.graph = build_visual_graph(exploration_observation, neighbors, distances)
            pickle.dump(self.graph, open("graph.pkl", "wb"))
        else:
            print("Loaded graph from graph.pkl")

        # Finally, build the netvlad lookup dictionary (for A* heuristic).
        if self.netvlad_lookup is None:
            self.netvlad_lookup = dict(zip(exploration_observation, self.database))
            print("Populated self.netvlad_lookup for A* heuristic.")

    def pre_navigation(self):
        """ Called once the exploration phase ends, to set up all data for pathfinding. """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Displays the next best view by running A* from the current FPV's closest neighbor
        to each of our goal images. Picks whichever path is shortest.
        """
        # 1) Find the closest image node in the graph to our current FPV
        curr_idx = self.get_neighbor(self.fpv)
        if curr_idx not in self.graph:
            print(f"Current FPV index '{curr_idx}' not in graph.")
            return curr_idx

        print(f"Current FPV index: {curr_idx}")

        # 2) Optionally check reachable nodes from curr_idx
        reachable_nodes = nx.descendants(self.graph, curr_idx) | {curr_idx}
        print(f"Reachable nodes from {curr_idx}: {len(reachable_nodes)}")

        best_next_idx = None
        best_path_len = float('inf')
        best_path = None

        # 3) For each goal view, run A* from curr_idx to that goal
        for goal_img in self.goal:
            goal_idx = self.get_neighbor(goal_img)
            if goal_idx not in self.graph:
                print(f"Goal '{goal_idx}' not in graph.")
                continue

            try:
                path = self.run_a_star(curr_idx, goal_idx)
                if len(path) > 1 and len(path) < best_path_len:
                    best_path_len = len(path)
                    best_next_idx = path[1]
                    best_path = path

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"No valid path from {curr_idx} to {goal_idx}")
                continue

        # 4) If we have no best_next_idx, no reachable goals were found
        if best_next_idx is None:
            print("No reachable goals found from current location.")
            return curr_idx

        # 5) Display the result
        print(f"Best path found: {best_path}")
        print(f"Next best index: {best_next_idx}")

        # Remove ".jpg" to pass to display_img_from_id
        next_id_stripped = best_next_idx.split(".")[0]
        self.display_img_from_id(next_id_stripped, "KeyboardPlayer:next_best_view")
        return best_next_idx

    def load_cleaned_filenames(self, json_path="data/data_info_cleaned.json"):
        """ Example helper to load 'approved' images from a JSON. """
        import json
        with open(json_path) as f:
            data = json.load(f)
        return {entry["image"] for entry in data}  # set of filenames

    def see(self, fpv):
        """
        Called by the environment every step with the current FPV image.
        We update self.fpv and optionally show the next best view (Q key).
        """
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                # We could do something during exploration, but it's provided
                pass
            elif self._state[1] == Phase.NAVIGATION:
                # If no goals yet, set them from environment
                if self.goal is None:
                    targets = self.get_target_images()
                    self.goal = targets

                # Press 'q' to display next best view
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
