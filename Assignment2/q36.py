import numpy as np
import matplotlib.pyplot as plt


def node_probability( dict , k ):
	adjacency = dict
	nodes = len( adjacency.keys() )
	A = np.zeros( ( nodes , nodes ) )
	for i in range( nodes ):
		for adj_node in adjacency[ i ] :
			A[ i ][ adj_node ] = 1
	#####################################################
	######PLEASE VERIFY#################################
	#####################################################
	d = np.sum( A , axis = 1 ) / ( np.sum( A ) )
	return d


def make_barabasi_albert_model( m , n , initial_dict ):
	"""
		Creates Barabasi-Albert preferential attachment random graph.
		Input: m = initial number of nodes
			   n = final number of nodes
			   initial dictionary of adjacency list of m nodes
		Return: returns adjacency dictionary
	"""
	for k in range( m , n  ):
		if k == 100 or k == 1000 or k == 10000 or k == 2000:
			degree_dist = degree_distribution( initial_dict , k )
			plot_degree_dist( m , initial_dict , k )
			print( assortativity( initial_dict , k ) , k)
		choices = np.random.choice( [i for i in range( k )] , ( m,) , p = node_probability( initial_dict , k ) , replace = False)
		initial_dict = make_new_adjacency_dict( initial_dict , choices , k )
	return initial_dict

# def uniform_random_growing( m , n , initial_dict ):
# 	for k in range( m , n  ):
# 		if k == 100 or k == 1000 or k == 10000:
# 			degree_dist = degree_distribution( initial_dict , k )
# 			plot_degree_dist( initial_dict , k )
# 			print( assortativity( initial_dict , k ) , k)
# 		choices = np.random.choice( [i for i in range( k )] , ( m,) , p = np.ones( ( k ,) ) / k , replace = False)
# 		initial_dict = make_new_adjacency_dict( initial_dict , choices , k )
# 	return


def make_new_adjacency_dict( new_dict , choices , k ):
	"""
		Creates new adjacency dicitionary for time step k.
		Input: Initial dictionary
			   choices = list of new connections at time step k
			   k = time step k
		Return: returns New adjacency dictionary
		"""
	new_dict[ k ] = list(choices)
	for choice in choices:
		new_dict[choice].append( k ) 
	return new_dict

def assortativity( initial_dict , nodes ):
	"""
		Finds assortativity of the graph
		Input: initial_dict = adjacency dictionary
				nodes = number of nodes at time step k
		Return: returns Assortativity (float)
		"""
	adjacency = initial_dict
	max_degree = 0
	total_links = 0
	A = np.zeros( ( nodes , nodes ) )

	for i in range( nodes ):
		for adj_node in adjacency[ i ] :
			A[ i ][ adj_node ] = 1

	d = np.matmul( A , np.ones( nodes ,  ) )
	# print(d)
	max_degree = int( np.amax( d ) )
	total_links = np.sum( d )
	M = np.zeros( ( max_degree  , max_degree  ) )
	for i in range( nodes ):
		for j in range( nodes ) :
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

def degree_distribution( initial_dict , nodes ):
	"""
		Find Degree Distribution dictionary
		Input: initial_dict = adjacency dictionary
				nodes = number of nodes at time step k
		Return: returns degree distribution dictionary
	"""	
	graph_dict = initial_dict
	degree_dist = {}
	for i in range( nodes ):
		if len( graph_dict[ i ] ) in degree_dist.keys():
			# print( len( self.__graph_dict[ i ] ) )
			degree_dist[ len( graph_dict[ i ] ) ] += 1 / nodes
		else:
			degree_dist[ len( graph_dict[ i ] ) ] = 1 / nodes
	for i in range( nodes ):
		if i in degree_dist.keys():
			pass
		else:
			degree_dist[ i ] = 0
	return degree_dist

	
def plot_degree_dist( m , adjacency_dict , nodes ):
	"""
		Plots the Barabsi Albert Graph degree distribution and the mean
		field degree distribution and the power-law distribution
		Input: 	m = initial number of nodes
				adjacency_dict = adjacency dictionary
				nodes = number of nodes at time step k
		Return: None
	"""	
    plt.close()
    num_nodes = nodes
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in range( num_nodes ):
        if len( adjacency_dict[n] ) > max_degree:
            max_degree = len( adjacency_dict[n] )
    # X-axis and y-axis values
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree+1):
        x.append(i)
        y_tmp.append(0)
        for n in range( num_nodes ):
            if len( adjacency_dict[n] ) == i:
                y_tmp[i] += 1
        y = [i/num_nodes for i in y_tmp]
    # Plot the graph
    plt.plot(x, y,label='Degree distribution',linewidth=0,marker = 'o' , markersize=3)
    # Check for the lin / log parameter and set axes scale
    w = [a for a in range(3,max_degree-15)]
    mean_field = []
    power_law = []
    for i in w:
        x = 2 * ( m**2 ) * (i**-3) * 1 # mean-field
        mean_field.append(x)
        x = i ** -2.5 # power-law
        power_law.append( x )

    plt.plot(w,mean_field, 'k-', color='#7f7f7f' , label="Mean Field")	
    plt.plot(w,power_law , label="Power law k^-2.5")	
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Degree distribution (log-log scale)')
    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.legend()
    plt.show()


def compute_page_rank( adjacency_dict , p = 0.15 ):
	"""
		Find Page Rank of the matrix using Google's Page rank algorithm
		Input: adjacency_dict = adjacency dictionary
				p = dampning fatctor, default = 0.15
		Return: returns degree distribution dictionary
	"""	
	nodes = len( list( adjacency_dict.keys() ) )
	A = np.zeros( ( nodes , nodes ) )
	for i in range( nodes ):
		for adj_node in adjacency_dict[ i ] :
			A[ i ][ adj_node ] = 1
	d = np.sum( A , axis = 1 )
	transistion_probability_matrix = np.divide( A.T , d )
	M = ( 1 - p ) * transistion_probability_matrix + p * np.ones( A.shape )/A.shape[0]
	eig_values , eig_vectors = np.linalg.eig( M.T )
	# eig_vectors /= eig_vectors[: , 0]
	print( eig_vectors[: , 0] , eig_values[0] )
	return eig_vectors[: , 0] , eig_values[0]

# ######################################################################################
# ___________________________MAIN CODE _______________________________________
# ######################################################################################

n = 101
m = 4
k = 4

initial_dict = {}
initial_dict = { 0 : [ 1 , 2 , 3 ] , 1: [ 0 , 2 , 3 ] , 2 : [ 0 , 1 , 3 ] , 3 : [ 0 , 1 , 2 ]  }

# make Barabsi-Albert preferential model
new_adjacency = make_barabasi_albert_model( m , n , initial_dict )
p = 0.15 # page rank damping factor
pi_inf , _ = compute_page_rank( new_adjacency , p ) # pi_inf is the stationary distribution



