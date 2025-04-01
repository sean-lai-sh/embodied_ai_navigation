# Example code modification for hyperparameter tuning
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Split data into training (80%) and validation (20%)
sift_descriptors = np.load("sift_descriptors.npy")
train_des, val_des = train_test_split(sift_descriptors, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_clusters': [64, 128, 256],
    'batch_size': [5000, 10000, 20000],
    'n_init': [5, 10]
}

best_inertia = float('inf')
best_params = {}

for n_clusters in param_grid['n_clusters']:
    for batch_size in param_grid['batch_size']:
        for n_init in param_grid['n_init']:
            print(f"Testing n_clusters={n_clusters}, batch_size={batch_size}, n_init={n_init}")
            
            # Train on training data
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                n_init=n_init,
                init='k-means++',
                verbose=0
            ).fit(train_des)
            
            # Evaluate on validation data
            val_inertia = kmeans.score(val_des) * -1  # Convert back to inertia
            print(f"Validation inertia: {val_inertia}")
            
            if val_inertia < best_inertia:
                best_inertia = val_inertia
                best_params = {
                    'n_clusters': n_clusters,
                    'batch_size': batch_size,
                    'n_init': n_init
                }

print(f"Best Params: {best_params}, Best Validation Inertia: {best_inertia}")