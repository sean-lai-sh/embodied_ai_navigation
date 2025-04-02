import os
import cv2
import pygame
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from natsort import natsorted
from vis_nav_game import Player, Action, Phase

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super().__init__()
        pygame.init()
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

        self.save_dir = "data/Images/"
        self.image_list = natsorted([f for f in os.listdir(self.save_dir) if f.endswith('.jpg')])
        self.sift = cv2.SIFT_create()

        # Load or compute components
        self._load_data()

        self.goal = None

    def _load_data(self):
        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        else:
            self.sift_descriptors = self._compute_sift_descriptors()
            np.save("sift_descriptors.npy", self.sift_descriptors)

        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        else:
            self.codebook = KMeans(n_clusters=64).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))

        if os.path.exists("vlad_db.pkl"):
            self.vlad_db = pickle.load(open("vlad_db.pkl", "rb"))
        else:
            self.vlad_db = self._compute_vlad_db()
            pickle.dump(self.vlad_db, open("vlad_db.pkl", "wb"))

        if os.path.exists("balltree.pkl"):
            self.balltree = pickle.load(open("balltree.pkl", "rb"))
        else:
            self.balltree = BallTree(self.vlad_db, leaf_size=40)
            pickle.dump(self.balltree, open("balltree.pkl", "wb"))

        if os.path.exists("graph.pkl"):
            self.graph = pickle.load(open("graph.pkl", "rb"))
        else:
            self.graph = self._build_knn_graph(k=5)
            pickle.dump(self.graph, open("graph.pkl", "wb"))

    def _compute_sift_descriptors(self):
        print("Extracting SIFT descriptors...")
        descriptors = []
        for filename in tqdm(self.image_list):
            img = cv2.imread(os.path.join(self.save_dir, filename))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                descriptors.extend(des)
        return np.array(descriptors)

    def _vlad(self, descriptors):
        pred = self.codebook.predict(descriptors)
        centers = self.codebook.cluster_centers_
        k = centers.shape[0]
        vlad = np.zeros((k, descriptors.shape[1]), dtype=np.float32)
        for i in range(k):
            if np.sum(pred == i) > 0:
                vlad[i] = np.sum(descriptors[pred == i] - centers[i], axis=0)
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad /= np.linalg.norm(vlad) + 1e-12
        return vlad

    def _compute_vlad_db(self):
        print("Computing VLAD descriptors...")
        database = []
        for filename in tqdm(self.image_list):
            img = cv2.imread(os.path.join(self.save_dir, filename))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                des = np.sqrt(des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-12))
                vlad_vec = self._vlad(des)
                database.append(vlad_vec)
        return np.array(database)

    def _build_knn_graph(self, k=5):
        print("Building k-NN graph...")
        G = nx.Graph()
        for i in range(len(self.vlad_db)):
            G.add_node(i)
        dists, indices = self.balltree.query(self.vlad_db, k=k+1)
        for i, (nbrs, ds) in enumerate(zip(indices, dists)):
            for j, d in zip(nbrs[1:], ds[1:]):
                G.add_edge(i, j, weight=d)
        return G

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return Action.QUIT
            elif event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            elif event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def get_neighbor(self, img):
        _, des = self.sift.detectAndCompute(img, None)
        if des is not None:
            des = np.sqrt(des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-12))
            vlad = self._vlad(des).reshape(1, -1)
            _, idx = self.balltree.query(vlad, k=1)
            return idx[0][0]
        return None

    def display_img(self, idx, title):
        filename = self.image_list[idx]
        img = cv2.imread(os.path.join(self.save_dir, filename))
        cv2.imshow(title, img)
        cv2.waitKey(1)

    def see(self, fpv):
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        pygame_img = pygame.image.frombuffer(fpv[:, :, ::-1].tobytes(), fpv.shape[1::-1], "RGB")
        self.screen.blit(pygame_img, (0, 0))
        pygame.display.flip()

        if self._state and self._state[1] == Phase.NAVIGATION:
            if self.goal is None:
                targets = self.get_target_images()
                self.goal = self.get_neighbor(targets[0])
                print(f"🎯 Goal image index: {self.goal}")
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.display_next_best_view()

    def display_next_best_view(self):
        src = self.get_neighbor(self.fpv)
        try:
            path = nx.shortest_path(self.graph, source=src, target=self.goal, weight="weight")
            if len(path) > 1:
                next_idx = path[1]
                print(f"Next best step: {next_idx}")
                self.display_img(next_idx, "Next Best View")
        except Exception as e:
            print("⚠️ No path found:", e)

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
