# Source: Spectral Clustering- William Fleshman
#         https://towardsdatascience.com/spectral-clustering-aba2640c0d5b

from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Spectral Clustering using Fiedler vector
fig, ( ax ) = plt.subplots(2, 2)
count = 1
noise_list = [ 0.1 , 0.2 , 0.3 , 0.4 ]
for i , noise in enumerate( noise_list ):
	# create the data
	X, labels = make_circles(n_samples=500, noise=noise, factor=.2)

	# use the nearest neighbor graph as our adjacency matrix
	A = kneighbors_graph(X, n_neighbors=5).toarray()
	# print(A)

	# create the graph laplacian
	D = np.diag(A.sum(axis=1))
	L = D-A

	# find the eigenvalues and eigenvectors
	vals, vecs = np.linalg.eig(L)

	# sort
	vecs = vecs[:,np.argsort(vals)]
	vals = vals[np.argsort(vals)]

	# use Fiedler value to find best cut to separate data
	clusters = np.array( vecs[:,1] > 0 , dtype = np.int )

	if i == 0 :
		ax[0][0].scatter( X[: , 0], X[: , 1] , c = clusters , marker = "*" )
		ax[0][0].set_title("Noise=" + str(noise))
	if i == 1 :
		ax[0][1].scatter( X[: , 0], X[: , 1] , c = clusters , marker = "*" )
		ax[0][1].set_title("Noise=" + str(noise))
	if i == 2 :
		ax[1][0].scatter( X[: , 0], X[: , 1] , c = clusters , marker = "*" )
		ax[1][0].set_title("Noise=" + str(noise))
	if i == 3 :
		ax[1][1].scatter( X[: , 0], X[: , 1] , c = clusters , marker = "*" )
		ax[1][1].set_title("Noise=" + str(noise))


fig.suptitle('Circles Data Spectral Clustering' )
plt.subplots_adjust( hspace=0.6)
plt.show()
