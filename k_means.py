import math

points = [(1,1), (2,1), (2,3), (3,2), (4,3), (5,5)]
centroids = [(1,1), (2,3)]
while True:
    clusters = [[],[]]
    for point in points:
        d1 = math.sqrt((point[0] - centroids[0][0])**2 + (point[1] - centroids[0][1])**2)
        d2 = math.sqrt((point[0] - centroids[1][0])**2 + (point[1] - centroids[1][1])**2)
        if d1 < d2:
            clusters[0].append(point)
        else:
            clusters[1].append(point)
    new_centroids = []
    for cluster in clusters:
        x = sum(point[0] for point in cluster)/len(cluster)
        y = sum(point[1] for point in cluster)/len(cluster)
        new_centroids.append((x,y))
    if new_centroids == centroids:
        break
    centroids = new_centroids
    print(f"Clusters:\n c1: {clusters[0]} \n c2: {clusters[1]}")
    print("Centroids:", centroids)

