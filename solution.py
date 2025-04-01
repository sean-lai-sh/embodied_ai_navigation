# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted

import logging

from FaiSS import FaissIndex
from graph import build_visual_graph
import networkx as nx
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained sift features and codebook
        self.sift_descriptors, self.codebook, self.faiss_index, self.graph = None, None, None, None
        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        if os.path.exists("faiss_index.pkl"):
            self.faiss_index = pickle.load(open("faiss_index.pkl", "rb"))
        if os.path.exists("graph.pkl"):
            self.graph = pickle.load(open("graph.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = None
        self.goal = None
        self.faiss_index = None
        self.graph = None

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist")

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        sift_descriptors = list()
        for img in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, img))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    # SIFT Gets us scale invariant (so featues can always be detected)
    # We store descriptors. But find why OpenCV is 128 Dimensional per keypoint (VERY VERY IMPORTANT )
    # Geneartes n by 128 which n is the num of images, so we have 3000 images
    
    def get_netVLAD(self, img):
        """
        Compute NetVLAD-style descriptor using SIFT descriptors and soft assignment.
        """
        _, des = self.sift.detectAndCompute(img, None)

        if des is None or len(des) == 0:
            return np.zeros(self.codebook.n_clusters * 128)

        centroids = self.codebook.cluster_centers_  # (K, D)
        k = centroids.shape[0]
        d = des.shape[1]

        # Soft assignment weights: inverse of distance to centroids (Gaussian-style)
        # Compute L2 distances between each descriptor and each centroid
        dists = np.linalg.norm(des[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
        sigma = 1e-6 + np.std(dists)  # Stability
        weights = np.exp(-dists**2 / (2 * sigma**2))  # (N, K)

        # Normalize weights so they sum to 1 across clusters
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        # Initialize NetVLAD descriptor
        netvlad_feature = np.zeros((k, d), dtype=np.float32)

        # Aggregate residuals weighted by soft assignment
        for cluster_id in range(k):
            soft_weights = weights[:, cluster_id][:, None]  # (N, 1)
            residuals = des - centroids[cluster_id]  # (N, D)
            weighted_residuals = soft_weights * residuals  # (N, D)
            netvlad_feature[cluster_id] = weighted_residuals.sum(axis=0)

        # Flatten and normalize
        netvlad_feature = netvlad_feature.flatten()

        # Power normalization (signed square root)
        netvlad_feature = np.sign(netvlad_feature) * np.sqrt(np.abs(netvlad_feature))

        # L2 normalization
        netvlad_feature = netvlad_feature / (np.linalg.norm(netvlad_feature) + 1e-12)

        return netvlad_feature

    def netvlad_aggregation(descriptors, centroids):
        """
        descriptors: (N, D) SIFT descriptors for one image
        centroids: (K, D) cluster centers from KMeans
        Returns: (K * D,) NetVLAD vector
        """
        K, D = centroids.shape
        N = descriptors.shape[0]

        # Assign descriptors to clusters (hard assignment here for simplicity)
        labels = np.argmin(np.linalg.norm(descriptors[:, None, :] - centroids[None, :, :], axis=2), axis=1)

        # Initialize NetVLAD vector
        vlad = np.zeros((K, D), dtype=np.float32)

        for i in range(K):
            if np.sum(labels == i) == 0:
                continue
            residuals = descriptors[labels == i] - centroids[i]  # (Ni, D)
            vlad[i] = residuals.sum(axis=0)

        # Flatten and normalize
        vlad = vlad.flatten()
        vlad = vlad / np.linalg.norm(vlad)  # L2 normalize the whole vector

        return vlad

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        fpv_desc = self.get_netVLAD(self.fpv)  # or get_VLAD
        curr_idx, _ = self.faiss_index.query(fpv_desc, k=1)
        curr_idx = curr_idx[0]
        return curr_idx

    def pre_nav_compute(self):
        """
        BuildGraph for A* Pathfinding
        """
        # Compute sift features for images in the database
        if self.sift_descriptors is None:
            print("Computing SIFT features...")
            self.sift_descriptors = self.compute_sift_features()
            np.save("sift_descriptors.npy", self.sift_descriptors)
        else:
            print("Loaded SIFT features from sift_descriptors.npy")
        
        if self.codebook is None:
            print("Computing codebook...")
            # TODO: Perform Hyper param tuning on cluster size, batch size, and number of iterations
            MiniBatchKMeans(
                n_clusters=128,
                batch_size=10_000,
                init='k-means++',
                n_init=5,
                verbose=1
            ).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            print("Loaded codebook from codebook.pkl")
        
        # NetVLAD aggregation for soft assignment since want approximations to create graphs
        if self.database is None:
            self.database = []
            print("Computing VLAD embeddings...")
            exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
            for img in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, img))
                VLAD = self.get_netVLAD(img)
                self.database.append(VLAD)

            # Build a FAISS index to enable graph-based navigation
            faiss_index = FaissIndex(dim=self.database[0].shape[0])
            faiss_index.build_index(self.database, image_ids=exploration_observation)  
            ## Pickle the FAISS index ####
            pickle.dump(faiss_index, open("faiss_index.pkl", "wb"))
            neighbors, distances = faiss_index.batch_query(self.database, k=5)
            ### build graph
            self.graph = build_visual_graph(exploration_observation, neighbors, distances, connect_temporal=False)    
            pickle.dump(self.graph, open("faiss_index.pkl", "wb")) 


    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        index = self.get_neighbor(self.fpv)
        # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        best_direction = None
        best_next_idx = None
        best_path_len = float("inf")
        print(index)
        print(self.goal)
        for direction, goal_idx in self.goal.items():
            try:
                path = nx.shortest_path(self.graph, source=index, target=goal_idx, weight='weight')
                if len(path) > 1 and len(path) < best_path_len:
                    best_path_len = len(path)
                    best_next_idx = path[1]
                    best_direction = direction
            except nx.NetworkXNoPath:
                continue  # Skip directions with no path

        if best_next_idx is not None:
            return best_next_idx, best_direction
        else:
            print("No valid next-best view found.")
            return index, None


    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())