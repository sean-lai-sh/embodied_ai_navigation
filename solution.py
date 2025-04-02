import os
import cv2
import numpy as np
import pygame
import pickle
import networkx as nx

from tqdm import tqdm
from natsort import natsorted
from vis_nav_game import Player, Action, Phase
from FaiSS import FaissIndex
from graph import build_visual_graph
from sklearn.cluster import MiniBatchKMeans


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super().__init__()
        pygame.init()

        self.screen = None
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

        self.fpv = None
        self.last_act = Action.IDLE

        self.save_dir = "./data/images_cleaned"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist. Please ensure exploration data is available.")

        self.sift = cv2.SIFT_create()
        self.sift_descriptors = None
        self.database = None

        self.codebook = pickle.load(open("codebook.pkl", "rb")) if os.path.exists("codebook.pkl") else None
        self.faiss_index = pickle.load(open("faiss_index.pkl", "rb")) if os.path.exists("faiss_index.pkl") else None
        self.graph = pickle.load(open("graph.pkl", "rb")) if os.path.exists("graph.pkl") else None

        self.goal = None

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            elif event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            elif event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def convert_opencv_img_to_pygame(self, img):
        img = img[:, :, ::-1]  # BGR to RGB
        return pygame.image.frombuffer(img.tobytes(), img.shape[1::-1], 'RGB')

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w = fpv.shape[:2]
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state and self._state[1] == Phase.NAVIGATION:
            if self.goal is None:
                targets = self.get_target_images()
                if targets and len(targets) == 4:
                    self.goal = {
                        "front": self.get_neighbor(targets[0]),
                        "right": self.get_neighbor(targets[1]),
                        "back": self.get_neighbor(targets[2]),
                        "left": self.get_neighbor(targets[3])
                    }
                    print(f"Set goal IDs: {self.goal}")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                next_idx, direction = self.display_next_best_view()
                if next_idx is not None:
                    self.display_img_from_id(next_idx, f"Next Best View: {direction}")

        img = self.convert_opencv_img_to_pygame(fpv)
        self.screen.blit(img, (0, 0))
        pygame.display.update()

    def get_netVLAD(self, img):
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.codebook.n_clusters * 128)

        centroids = self.codebook.cluster_centers_
        k, d = centroids.shape
        dists = np.linalg.norm(des[:, None] - centroids[None], axis=2)
        sigma = 1e-6 + np.std(dists)
        weights = np.exp(-dists**2 / (2 * sigma**2))
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        netvlad = np.zeros((k, d), dtype=np.float32)
        for i in range(k):
            soft_weights = weights[:, i:i+1]
            residuals = des - centroids[i]
            netvlad[i] = (soft_weights * residuals).sum(axis=0)

        netvlad = np.sign(netvlad) * np.sqrt(np.abs(netvlad))
        return netvlad.flatten() / (np.linalg.norm(netvlad) + 1e-12)

    def get_neighbor(self, img):
        vlad = self.get_netVLAD(img)
        idx, _ = self.faiss_index.query(vlad, k=1)
        return idx[0]

    def display_img_from_id(self, id, window_name):
        path = os.path.join(self.save_dir, str(id) + ".jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image ID {id} not found")

    def display_next_best_view(self):
        if self.goal is None or self.graph is None or self.faiss_index is None:
            print("Missing goal, graph, or index.")
            return None, None

        current_index = str(self.get_neighbor(self.fpv))
        best_path_len = float("inf")
        best_next_idx = None
        best_direction = None

        for direction, goal_idx in self.goal.items():
            try:
                path = nx.shortest_path(
                    self.graph, source=current_index, target=str(goal_idx), weight='weight'
                )
                if len(path) > 1 and len(path) < best_path_len:
                    best_path_len = len(path)
                    best_next_idx = path[1]
                    best_direction = direction
            except nx.NetworkXNoPath:
                continue

        return best_next_idx, best_direction

    def pre_nav_compute(self):
        if self.sift_descriptors is None:
            print("Extracting SIFT descriptors...")
            descriptors = []
            for fname in tqdm(natsorted(os.listdir(self.save_dir))):
                if not fname.endswith(".jpg"):
                    continue
                img = cv2.imread(os.path.join(self.save_dir, fname))
                _, des = self.sift.detectAndCompute(img, None)
                if des is not None:
                    descriptors.extend(des)
            self.sift_descriptors = np.array(descriptors)
            np.save("sift_descriptors.npy", self.sift_descriptors)

        if self.codebook is None:
            print("Building codebook...")
            self.codebook = MiniBatchKMeans(n_clusters=128, batch_size=10000).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))

        if self.database is None:
            self.database = []
            file_names = natsorted([f for f in os.listdir(self.save_dir) if f.endswith(".jpg")])
            for fname in tqdm(file_names, desc="VLAD encoding"):
                img = cv2.imread(os.path.join(self.save_dir, fname))
                self.database.append(self.get_netVLAD(img))

            self.faiss_index = FaissIndex(dim=self.database[0].shape[0])
            self.faiss_index.build_index(self.database, image_ids=file_names)
            pickle.dump(self.faiss_index, open("faiss_index.pkl", "wb"))

            neighbors, distances = self.faiss_index.batch_query(self.database, k=5)
            self.graph = build_visual_graph(file_names, neighbors, distances, connect_temporal=False)
            pickle.dump(self.graph, open("graph.pkl", "wb"))

    def pre_navigation(self):
        super().pre_navigation()
        self.pre_nav_compute()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
