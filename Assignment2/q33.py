import numpy as np
import matplotlib.pyplot as plt
import operator

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



def second_small_eig( A ):
	"""
		Find Second smallest eigen value
		Input: Adjacency matrix
		Return: returns Second smallest eigen value
		"""
	d = np.sum( A , axis = 1 )
	L = np.diag( d ) - A
	eig_values , eig_vectors = np.linalg.eig( L )
	copy_eigen_values = list(eig_values)
	copy_eigen_values.sort( )
	SS_eigen_value = copy_eigen_values[1]
	return SS_eigen_value/copy_eigen_values[-1]

# ######################################################################################
# ___________________________MAIN CODE _______________________________________
# ######################################################################################

n = 10
p = 0.3

# Plot the assortativity vs second smallest eigen value
assortativity_list = {}
for i in range (3):
	G = Graph()
	G.make_Erdos_Renyi_graph( n , p)
	assortativity_list[abs(G.assortativity()) ] = second_small_eig( G.adjacency_sparse() )


sorted_d = sorted(assortativity_list.items(), key=operator.itemgetter(1))
print(sorted_d)

assortativity_list = [ a for a , _ in sorted_d ]
second_small_eig_list = [ s for _ , s in sorted_d ]

plt.plot( second_small_eig_list , assortativity_list )
plt.title( "Assortativity vs Second smallest Eigen value" )
plt.xlabel( "Second smallest eigen value" )
plt.ylabel( "Assortativity" )
plt.show()