import numpy as np
from sklearn.cluster import KMeans

gfpf = np.load('.gfpf.npy')
km = KMeans(n_clusters=333, n_jobs=8).fit(gfpf)
print(km.labels_)

