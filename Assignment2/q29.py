import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Graph(  ):
	"""
		Creates Graph Object.
		Input:  (optional) graph_dict = adjacency dictionary
				(optional)n = number of nodes
		Return: returns graph object
		"""
	def __init__( self , graph_dict = None , nodes = 0 ):
		self.__graph_dict = graph_dict
		self.nodes = nodes

	def vertices ( self ):
		"""
		Return: returns Vertices
		"""
		return  list(self.__graph_dict.keys() ) 

	def edges ( self ):
		"""
		Return: returns Edges
		"""
		edges = set()
		for vertex in self.vertices( ):
			for edge_vertex in self.__graph_dict[ vertex ]:	
				if ( edge_vertex , vertex ) not in edges:
					edges.add( ( vertex , edge_vertex  ) )
		return edges

	def adjacency_dense( self ):
		return self.__graph_dict

	def adjacency_sparse( self ):
		"""
		Creates sparse Adjacency matrix.
		Input: adjacency dictionary
		Return: returns Adjacency matrix
		"""
		adjacency = self.__graph_dict
		A = np.zeros( ( self.nodes , self.nodes ) )

		for i in range( self.nodes ):
			for adj_node in adjacency[ i ] :
				A[ i ][ adj_node ] = 1
		return A
		
	def make_Erdos_Renyi_graph( self ,  nodes , probability ):
		"""
		Creates Erdos-Renyi random graph.
		Input: n = number of nodes
			   probablilty = probability of edge present
		Return: returns None
		"""
		self.__graph_dict = {}
		if self.nodes == 0:
			self.nodes = nodes
		for i in range( self.nodes ):
			if i not in self.__graph_dict.keys():
				self.__graph_dict[ i ] = []
			for j in range( self.nodes ):
				if j not in self.__graph_dict.keys():
					self.__graph_dict[ j ] = []
				if ( np.random.choice( [0 , 1] , (1,) ,p= [ 1 - probability / 2 , probability / 2 ] )[ 0 ] == 1 ) and i != j:
					# Edge Present
					if i not in self.__graph_dict[j] and j not in self.__graph_dict[i]:
						self.__graph_dict[i].append( j )
						self.__graph_dict[j].append( i )
				else:
					pass


	def degree_distribution( self ):
		"""
		Find Degree Distribution dictionary
		Input: None or graph object
		Return: returns degree distribution dictionary
		"""
		degree_dist = {}
		for i in range( self.nodes ):
			if len( self.__graph_dict[ i ] ) in degree_dist.keys():
				# print( len( self.__graph_dict[ i ] ) )
				degree_dist[ len( self.__graph_dict[ i ] ) ] += 1 / self.nodes
			else:
				degree_dist[ len( self.__graph_dict[ i ] ) ] = 1 / self.nodes
		for i in range( self.nodes ):
			if i in degree_dist.keys():
				pass
			else:
				degree_dist[ i ] = 0
		return degree_dist

	def assortativity( self  ):
		"""
		Finds assortativity of the graph
		Input: None or Graph object
		Return: returns Assortativity (float)
		"""
		adjacency = self.__graph_dict
		max_degree = 0
		total_links = 0
		
		A = np.zeros( ( self.nodes , self.nodes ) )

		for i in range( self.nodes ):
			for adj_node in adjacency[ i ] :
				A[ i ][ adj_node ] = 1

		d = np.matmul( A , np.ones( self.nodes ,  ) )
		# print(d)
		max_degree = int( np.amax( d ) )
		total_links = np.sum( d )
		M = np.zeros( ( max_degree  , max_degree  ) )
		for i in range( self.nodes ):
			for j in range( self.nodes ) :
				if A[ i ][ j ] == 1:
					M[ int(d[i] -1 ) ][ int(d[j] - 1 ) ] += 1
		M = M / total_links
		q = np.matmul( M , np.ones( M.shape[0] ,  ) )
		# q = np.matmul( np.ones( ( M.shape[0]) ) , M )

		ijM = [ [ i * j * M[ i - 1 ][ j - 1 ] for j in range(1 , M.shape[0] ) ] for i in range(1 , M.shape[1] ) ]
		iq = [ i * q[ i - 1] for i in range( 1 , q.shape[0] ) ]
		i_2_q = [ i * i * q[ i - 1 ] for i in range( 1 , q.shape[0] ) ]
		sum_M = np.sum( ijM )
		sum_q = np.sum( iq )
		sum_q_squared = np.sum( i_2_q )
		assortativity = ( sum_M  - sum_q * sum_q ) / ( sum_q_squared - sum_q * sum_q )
		return assortativity


def fit_C_D( degree_dist ):
	"""
		Fit C and D to the equation Ck - Dklogk
		Input: Degree distribution dictionary
		Return: returns (coeffiicent_of_determination, (C , D))
		"""
	y = []
	x_train = []
	for i in range( 1 , len( degree_dist.keys() ) ):
		if degree_dist[ i 	] != 0 :
			y.append( degree_dist[ i - 1]  )
			x_train.append( [ i , -i * np.log( i ) ] )
	reg = LinearRegression().fit( x_train , y )
	reg_score2 = reg.score( x_train , y )
	return reg.score( x_train , y ) , reg.coef_  


def freeman_centrality( G ):
	"""
		Freeman Centrality
		Input: Graph Object
		Return: returns Freeman Centrality (Float)
		"""
	adjacency = G.adjacency_dense()
	max_degree = 0
	total_links = 0
	
	for i in range( G.nodes):
		degree = len( adjacency[i] )
		total_links += degree
		if max_degree < degree :
			max_degree = degree 
	centrality = 0
	for i in range( G.nodes ):
		centrality += max_degree - len( adjacency[i] )
	return centrality / ( ( G.nodes - 1 ) * ( G.nodes - 2 ) )

# ######################################################################################
# ___________________________MAIN CODE _______________________________________
# ######################################################################################

n = 100
p = 0.3

G = Graph()
G.make_Erdos_Renyi_graph( n , p) # Creates Erdos Renyi Graph
# ___________________________Part A _________________________________________

d = G.degree_distribution() 
# Plot Degree Distribution
dd = []
d_x = []
for i in range( n ) :
	if i in d.keys():
		d_x.append( i )
		dd.append( d[i] )
	else:
		dd.append(0)
plt.plot( d_x , dd )
plt.xlabel( "k ( Degree )" )
plt.ylabel( "Degree Distribution rho_k ( Count )" )
plt.title( "Degree Distribution of Erdos-Renyi Graph" )
plt.xscale("log")
plt.yscale("log")
plt.show()

# _____________________Part B_________________________

# Find C and D
print( fit_C_D( G.degree_distribution() ) )

# _______________Part C_______________________________________________
# Check C and D for various values of probability

p_list = np.arange( 0.01 , 1.01 , 0.1 )
coef_determination = []
C_list = []
D_list = []

for p in p_list :
	avg_determination = 0
	avg_C = 0
	avg_D = 0
	for i in range(5):
		G = Graph()
		G.make_Erdos_Renyi_graph( n , p )
		avg_determination += fit_C_D( G.degree_distribution() )[0]
		avg_C += fit_C_D( G.degree_distribution() )[1][0]
		avg_D += fit_C_D( G.degree_distribution() )[1][1]
	avg_determination /= 5
	avg_C /= 5
	avg_D /= 5
	coef_determination.append( avg_determination )
	C_list.append( avg_C )
	D_list.append( avg_D )
plt.plot( p_list , coef_determination , label = "Coefficient of Determination")
plt.plot( p_list , C_list , label= "C")
plt.plot( p_list , D_list , label = "D" )
plt.legend()
plt.xlabel( "probability p" )
plt.ylabel( "Coefficient of Determination" )
plt.title( "" )
# plt.show()


#________________Part D_____________________________________________________________
# Plot the assortativity for various values of n and probability

p_list = np.arange( 0.1 , 1.01 , 0.1 )
n_list = [ 10 ,25 , 50 , 75 , 100 ]
plt.figure()

count = 1
for n in n_list:
	assortativity_list = []
	for p in p_list :
		G = Graph()
		G.make_Erdos_Renyi_graph( n , p )
		assortativity_list.append( G.assortativity() )
	plt.subplot( len(n_list) , 1 , count )
	plt.plot( p_list , assortativity_list )
	plt.title( "n=%d" %( n ) )
	plt.xlabel( "probability p" )
	plt.ylabel( "Assortativity" )
	count += 1
plt.subplots_adjust( hspace=2.5)
plt.show()


# _______________________________________Part E ______________________________________________
# Find out Freeman's centrality for various values of probability

p_list = np.arange( 0. , 1.01 , 0.05 )
centrality = []
for p in p_list :
	G = Graph()
	G.make_Erdos_Renyi_graph( n , p )
	centrality.append( freeman_centrality( G ) )

plt.plot( p_list , centrality )
plt.xlabel( "probability p" )
plt.ylabel( "Freeman's Centrality" )
plt.title( "Freeman's Centrality as it varies with p " )
plt.show()
# ____________________________________________________________________________________________
