import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.image_ids = []  # optional: to track which image each vector came from

    def build_index(self, descriptors, image_ids=None):
        """
        descriptors: (N, D) numpy array of NetVLAD descriptors
        image_ids: list of image filenames (optional)
        """
        descriptors = np.array(descriptors, dtype='float32')
        self.index.add(descriptors)

        if image_ids is not None:
            self.image_ids = image_ids

    def query(self, query_descriptor, k=5):
        """
        Returns top-k indices (and distances) for a query NetVLAD descriptor.
        """
        query_descriptor = np.array([query_descriptor], dtype='float32')  # shape: (1, D)
        distances, indices = self.index.search(query_descriptor, k)
        return indices[0], distances[0]

    def batch_query(self, query_descriptors, k=5):
        """
        Batch top-K queries for all descriptors.
        Returns: (N, k) indices and (N, k) distances
        """
        query_descriptors = np.array(query_descriptors, dtype='float32')
        D, I = self.index.search(query_descriptors, k)
        return I, D
