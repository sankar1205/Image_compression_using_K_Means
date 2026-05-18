import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

org_img = plt.imread('/content/bird_small.png')

plt.imshow(org_img)

x_img = np.reshape(org_img, (org_img.shape[0]*org_img.shape[1],3))

k = 16
kmeans = KMeans(n_clusters=k,random_state=0)
kmeans.fit(x_img)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
closest_cluster = kmeans.predict(x_img)

plt.scatter(x_img[:, 0], x_img[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print(silhouette_samples(x_img,labels))

x_rec = centers[closest_cluster, :]
x_rec = np.reshape(x_rec, org_img.shape)
print(closest_cluster)

plt.imshow(x_rec)

