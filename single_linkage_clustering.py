data = [18, 22, 25, 42, 27, 43]
clusters = [[x] for x in data]
c=1
while len(clusters) > 1:
    min_dist = float('inf')    
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for a in clusters[i]:
                for b in clusters[j]:
                    d = abs(a - b)
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)
    i,j = pair
    clusters[i] += clusters[j]
    del clusters[j]
    print(f"Merged cluster {c} : {clusters[i]} with distance {min_dist}")
    c += 1