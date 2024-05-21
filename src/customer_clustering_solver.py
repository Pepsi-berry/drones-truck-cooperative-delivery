import numpy as np
import matplotlib.pyplot as plt


def kmeans_with_fixed_centroids(X, k, fixed_centroids, max_iters=200):
    num_fixed = len(fixed_centroids)
    
    remaining_indices = np.random.choice(X.shape[0], k - num_fixed, replace=False)
    centroids = np.vstack((fixed_centroids, X[remaining_indices]))
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # print(distances)
        
        clusters = np.argmin(distances, axis=1)
        # print(clusters)
        
        for i in range(num_fixed, k):
            if np.any(clusters == i):
                centroids[i] = X[clusters == i].mean(axis=0)
    
    return clusters, centroids


def constrains(labels, centroids, nodes, limit=1_500):
    distances = np.linalg.norm(nodes - centroids[labels], axis=1)
    
    if (distances < limit).all():
        return True
    else:
        return False


def customer_cluster(nodes, fixed_centroids):
    num_nodes = nodes.shape[0]
    # num_fixed_centroids = fixed_centroids.shape[0]
    for i in range(int(num_nodes / 4), num_nodes):
        # keep the number of clusters small
        for _ in range(2):
            while True:
                break_condition = True
                labels, centroids = kmeans_with_fixed_centroids(nodes, i, fixed_centroids)
                
                # avoid empty clustering
                for k in range(i):
                    if not np.isin(k, labels):
                        break_condition = False
                        break
                if break_condition:
                    break
            if constrains(labels, centroids, nodes):
                return labels, centroids
    
    return None, None


def solve_drones_truck_with_parking(customer_points, customer_truck_points, grid_size=125):
    labels, centroids = customer_cluster(customer_points, customer_truck_points)
    
    # the center point obtained through k_means is not necessarily on the road, 
    # transfer to its corresponding parking point on the road
    for centroid in centroids:
        bias_x = centroid[0] % grid_size
        bias_y = centroid[1] % grid_size
        
        if bias_x == 0 or bias_y == 0:
            continue
        
        bias_x = min(bias_x, grid_size - bias_x)
        bias_y = min(bias_y, grid_size - bias_y)
        
        if bias_x < bias_y:
            centroid[0] = round(centroid[0] / grid_size) * grid_size
        else:
            centroid[1] = round(centroid[1] / grid_size) * grid_size
    
    clusters = [
        [] for _ in range(max(labels) + 1)
    ]
    for idx, label in enumerate(labels):
        clusters[label].append(idx + 1)
            
    return labels, clusters, centroids


if __name__ == "__main__":
    # X = np.array([[1, 2], [1, 4], [1, 0],
    #             [4, 2], [4, 4], [4, 0],
    #             [0, 1], [5, 2], [5, 3]])

    # fixed_centroids = np.array([[1, 2], [4, 4]])

    X = np.random.randint(5_000, size=(20, 2))
    # print(X)
    
    fixed_centroids = X[:4]
    
    k = 5
    
    # clusters, centroids = kmeans_with_fixed_centroids(X, k, fixed_centroids)
    # clusters, centroids = customer_cluster(X, fixed_centroids)
    
    labels, clusters, centroids = solve_drones_truck_with_parking(X, fixed_centroids)
    
    print("cluster labels:", labels)
    print("clusters: ", clusters)
    print("centroids:", centroids)
    
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()
