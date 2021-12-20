# Databricks notebook source
# MAGIC %md # HW 5 - Page Rank
# MAGIC ### Team 8: Yao Chen, Toby Petty, Ferdous Alam, Zixi Wang
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2021`__
# MAGIC 
# MAGIC 
# MAGIC In Weeks 8 and 9 you discussed key concepts related to graph based algorithms and implemented SSSP.   
# MAGIC In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia.
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions under which it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph in order to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__ 

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.   

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# RUN THIS CELL AS IS. 
tot = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
for item in dbutils.fs.ls(DATA_PATH):
  tot = tot+item.size
tot
# ~4.7GB

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md # Question 1: Distributed Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concerns that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading and your async lectures. 
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example? 
# MAGIC 
# MAGIC * __b) short response:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Do not respond in terms of any specific algorithm. Think in terms of the nature of the graph datastructure itself).*
# MAGIC 
# MAGIC * __c) short response:__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC 
# MAGIC * __d) short response:__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ For example, Facebook (meta) dataset would be appropriate to represent as a graph. Each user's Facebook profile is the node and his/her connections are the edges.This type of graph is 'undirected' since each pair of friendship connections are mutal relationships - both users know each other. This is a two-way connection, which can be different from Twitter where a follower connection is directed and doesn't need to be connected mutually. The average "in-degree" of a node will be the average number of connections for a single user profile.
# MAGIC 
# MAGIC > __b)__ The data structure of graphs require a large number of iterations to work with. However, a map-reduce paradigm is not suitable for iterative computing because it can only conduct one iteration. To implement an iterative algorithm, the programmer needs to explicitly handle the iteration. For example, after each iteration, the programmer needs to write the intermediate results into a specified location for the next iteration (another map-reduce procedure) to read. Therefore, implementing map-reduce paradigm is difficult on graphs because such implementation involves heavy I/O and extra job start up time, which is very inefficient.
# MAGIC 
# MAGIC > __c)__ Dijkstra's algorithm is an algorithm for finding the shortest paths from a single source node to all other nodes in a graph. Since this algorithm uses a data structure for storing and querying partial solutions sorted by distance from the source node by using a minimum priority queue, it is hard to parallelize as in Map-Reduce we cannot maintain the state of the min priority queue. For example, in the reducer phase, if distance to a node has gotten shorter, we need to recompute the shortest path for each node in the subgraph emanating from that node.
# MAGIC 
# MAGIC > __d)__ In order to parallelize using Breadth-First-Search, we need to put source node in frontier and go through each potential solution to find the shortest path, which can be computationally expensive. For each node, in the Map phase we need to emit a key value pair for each neighbor in the adjacency list; in the Reduce phase, we can then select the minimum value for keys which is the shortest distance.

# COMMAND ----------

# MAGIC %md # Question 2: Representing Graphs
# MAGIC 
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of $n_1$, $n_2$, etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.
# MAGIC 
# MAGIC * __b) short response:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code to complete the function `get_adj_list()`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2 Student Answers:
# MAGIC > __a)__ The figure in 5.1 has 5 vertices and 9 edges.
# MAGIC Dense graphs are densely connected where the number of edges is usually `O(n^2)` n being the number of vertices.
# MAGIC For the example this not being the case, appears to be sparse. In the graph there are a lot of zeros in the matrix, meaning that there are few edges between the nodes. The denser the matrix, the more edges/connections we will see between the nodes in a graph, the fewer zeros there will be in the adjacency matrix, the more neighbors each node will have in the adjacency list.
# MAGIC From a storage perspective an adjacency matrix requires  `O(V^2)` space whereas an adjacency list requires `Θ(|V|+|E|)` space to store.
# MAGIC So for sparse graph using an adjacency matrix is wasteful from a space perspective but quering it will be faster than an adjacency list, it can be implemented in `O(|V|+|E|log|E|)` time.
# MAGIC If the graph is dense both representations will need a quadratic amount of space to store, therefore using the adjacency matrix for `O(1)` edge queries is the better option.
# MAGIC 
# MAGIC > __b)__ The given code for the toy graph is an example of a directed graph since it has edges with direction and there are values on both sides of the diagonal line of the matrix. 
# MAGIC The adjacency matrix will be different for a directed graph and undirected graph because for an undirected graph the matrix is symmetric i.e.: `a_i,j=a_j,i` for every pair `i,j`, or `A=A^T`. For a directed graph there is an edge from `i` to `j` but necessarily an edge from `j` to `i` which would imply that the adjacency matrix of an directed graph is will not necessarily be symmetric i.e `A<>A^T`.

# COMMAND ----------

# part a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
      adj_matr.at[edge[0],edge[1]]=1
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
      adj_list[edge[0]].append(edge[1])
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md # Question 3: Markov Chains and Random Walks
# MAGIC 
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC * __b) short response:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC 
# MAGIC * __c) short response:__ A Markov chain consists of $n$ states plus an $n\times n$ transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?
# MAGIC 
# MAGIC * __d) code + short response:__ What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC 
# MAGIC * __e) code + short response:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ In the analogy of an infinite web-surfer, the PageRank metric represents the frequency with which the surfer will visit a page relative to all others in the graph. It is a probability distribution over all the nodes in the graph (i.e. all the pages on the web), giving the likelihood that a random walk across the graph will arrive at a page. As PageRank algorithm reaches its steady state, the most important node (with highest frequency) will be the one that is the most likely for the web surfer to land on.
# MAGIC 
# MAGIC > __b)__ The Markov Property is a property of systems whereby the current state of the system gives all the relevant information required, meaning the full history of how the system arrived at the current state is not needed. An example would be a chess game, where the current state of the board (plus whose turn it is) is the only information required to analyse the game and decide the best next move; knowing the full history of how the game arrived in that state is not needed. In the context of the PageRank algorithm, the Markov Property means that on each iteration of the algorithm we only need to know the current PageRank values of each node in the graph in order to proceed with the next iteration of the algorithm; we don't need the full history of how the PageRank values have changed in all previous iterations.
# MAGIC 
# MAGIC > __c)__ In the context of PageRank, the $n$ states of the Markov Chain correspond to the individual webpages, which means that the transition matrix $P$ has dimensions $nxn$, where each entry $P_{ij}$ gives the probability of a surfer at a webpage denoted by row index $i$ transitioning to a webpage denoted by column index $j$. Given that the number of webpages is likely on the order of billions the implications are that this is an extremely large matrix.
# MAGIC 
# MAGIC > __d)__ The "right stochastic matrix" is the transition probability matrix in the Markov Chain which gives the probabilities of moving from each page in the index to each page in the columns. It must be a square matrix and each row in the matrix must sum to 1 to be a valid probability distribution.
# MAGIC 
# MAGIC > __e)__ When we run the `power_iteration` function for longer than 10 times we find that it eventually converges to the steady state probabilities after <b>52</b> iterations. The most central/highest ranked node is node <b>E</b>, which makes sense intuitively as it is the only node which is directly connected to all other nodes in the toy graph.

# COMMAND ----------

# part d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
transition_matrix = TOY_ADJ_MATR.divide(TOY_ADJ_MATR.sum(axis = 1), axis = 0).fillna(0)
################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing initial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################
    
    if verbose:
      print("|  ".join([s.ljust(12) for s in ["Iter"] + list(tMatrix.columns)]))

    state_vector = xInit.reshape(5,1)
    for i in range(nIter):
      state_vector = np.matmul(tMatrix.values.T, state_vector)

      if verbose:
        print("|  ".join([s.ljust(12) for s in [f"{i+1}"] + [f"{s[0].round(8)}" for s in state_vector]]))

    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 10, verbose = True)

# COMMAND ----------

steady_states = power_iteration(xInit, transition_matrix, 52, verbose = False)
steady_states

# COMMAND ----------

# MAGIC %md __`Expected Output for part e:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC Node A: 0.10526316  
# MAGIC Node B: 0.15789474  
# MAGIC Node C: 0.18421053  
# MAGIC Node D: 0.23684211  
# MAGIC Node E: 0.31578947  
# MAGIC ```

# COMMAND ----------

# MAGIC %md # Question 4: Page Rank Theory
# MAGIC 
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' in Question 3 and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from question 3 on a 'not nice' graph.
# MAGIC 
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) code + short response:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC 
# MAGIC * __b) short response:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __d) short response:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __e) short response:__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC > __a)__ The state vector doesn't sum to 1 at later steps, which means the stochastic property is violated. Therefore, it is not converging. Node E is a "dangling node which only has incoming edges, therefore it is acting as sink for the mass being passed to it.
# MAGIC 
# MAGIC > __b)__ E is the dangling node in this graph. With a dangling node, there will be a row of all zeros in the transition matrix. Every time we apply the transition matrix, there will be a state probability equal to zero and thus the sum of state probability is not 1. To solve this, we can flip a biased coin. If it is head, we keep the calculated state vector; otherwise, we reset the state vector by redistributing the probability equally to each state.
# MAGIC 
# MAGIC > __c)__ Being irreducible means there is a path from every node to every other node. No, the webgraph isn't naturally irreducible by itself since a link is usually established if two pages are related. There are no links between irrelevant pages so it is not irreducible.
# MAGIC 
# MAGIC > __d)__ Being aperiodic means greatest common divisor of all cycle length is 1. Yes, in the case of webgraph, there will be some dangling nodes that have self-loops. Self-loop's cycle length is 1 and thus the greatest common divisor has to be 1. Therefore, webgraph is aperiodic.
# MAGIC 
# MAGIC > __e)__ Teleportation is introduced to ensure aperiodicity and irreducibility. In the case of a random surfer, it means there is a chance that the surfer will ignore the page rank algorithm and randomly choose any node in the graph as the next step. This is the ability to jump to any pages from any current page. The random surfer can transition to any new url from current website. PageRank can redistribute the probability mass from dangling pages to all the other pages.

# COMMAND ----------

# part a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################
adj_matrix = get_adj_matr(TOY2_GRAPH)
trans_matrix = adj_matrix.divide(adj_matrix.sum(axis = 1), axis = 0).fillna(0)
power_iteration(xInit, trans_matrix, 10, verbose = True)
####### (END) YOUR CODE #################

# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC 
# MAGIC As in previous homeworks we'll be using a 2GB subset of this data, which is available to you in this dropbox folder: 
# MAGIC > https://www.dropbox.com/sh/2c0k5adwz36lkcw/AAAAKsjQfF9uHfv-X9mCqr9wa?dl=0. 
# MAGIC 
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation(note that we'll use the 'indexed out' version of the graph) and to take a look at the files.

# COMMAND ----------

dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/test_graph.txt', "r") as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(10)

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: EDA part 1 (number of nodes)
# MAGIC 
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data anlysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) short response:__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC 
# MAGIC * __b) code + short response:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC 
# MAGIC * __c) code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC 
# MAGIC * __d) short response:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC > __a)__ The raw data is wiki pages represented as adjacency list in txt format. The first value is the node and the second part are the linked nodes (as keys) with the number of links to those pages (as values). Each record is a key-value pair where key is the wiki page ID (node) and value is a dictionary (adjacency list) showing the neighbors/linked page IDs along with weights. The weights can be interpreted as the number of times a page is linked/referenced in the source page.
# MAGIC 
# MAGIC > __b)__ The raw dataset file in txt format only contains non-dangling nodes. The dangling nodes don't have a linked page that are present in the adjacency list so they will have empty record of adjacency list. To get total number of nodes, we need to go through each node and get its linked node as well. Then we can find the total number by counting the number of distinct node ID.
# MAGIC 
# MAGIC > __d)__ The total number of dangling nodes are number of nodes minus number of records, since dangling nodes are in the adjacency list but do not have their own record. Therefore the calculation is 15,192,277 - 5,781,290 = 9,410,987 dangling nodes.

# COMMAND ----------

# part b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290
print(wikiRDD.count())

# COMMAND ----------

# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############
    def get_nodes(line):
      node, adj_node = line.split('\t')
      adj_node = list(ast.literal_eval(adj_node).keys())      
      for n in adj_node + [node]:
        yield n
    totalCount = dataRDD.flatMap(get_nodes)\
                        .distinct()\
                        .count()
                        
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# part c - run your counting job on the test file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(testRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part c - run your counting job on the full file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(wikiRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# MAGIC %md # Question 6 - EDA part 2 (out-degree distribution)
# MAGIC 
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC  * count the out-degree of each non-dangling node and return the names of the top 10 pages with the most hyperlinks
# MAGIC  * find the average out-degree for all non-dangling nodes in the graph
# MAGIC  * take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC  
# MAGIC  
# MAGIC * __b) short response:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ In the context of the PageRank algorithm, for each node the PageRank probability mass needs to be distributed to its neighbour nodes in the adjacency list. Each node's out-degree is used to compute how much of the PageRank probability mass will need to be passed along to its outgoing edges.
# MAGIC 
# MAGIC > __c)__ If a node's out-degree is 0, that means this is a dangling node that doesn't have outgoing links to other nodes. It makes the graph "not nice" and reducible since the total probability mass will not be conserved. In the mapper phase, there would be no key-value pairs emitted for a dangling node. Therefore we will not see convergence to steady state probabilities among nodes. Noted the total PageRank values of all nodes should sum up to 1. In PageRank, we can redistribute the probability mass for dangling nodes among all nodes evenly. Whenever the mapper encounters a node with an empty adjacency list, we can use a counter to keep track of the dangling node's PageRank value (probability mass). At the end of the iterations, we can check the counter to figure out the lost of probability mass due to dangling nodes and then redistribute it to all nodes.

# COMMAND ----------

# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############
    top, avgDegree, sampledCounts = None, None, None
    cachedRDD = dataRDD.map(parse)\
                       .mapValues(lambda x: len(x))\
                       .cache()
    top = cachedRDD.takeOrdered(10, key=lambda x:-x[1])
    cached_RDD_new = cachedRDD.map(lambda x : x[1]).cache()
    avgDegree = cached_RDD_new.mean()
    sampledCounts = cached_RDD_new.takeSample(False, n)
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# part a - run your job on the test file (RUN THIS CELL AS IS)
start = time.time()
test_results = count_degree(testRDD,10)
print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])

# COMMAND ----------

# part a - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part a - run your job on the full file (RUN THIS CELL AS IS)
start = time.time()
full_results = count_degree(wikiRDD,1000)

print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part a - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md # Question 7 - PageRank part 1 (Initialize the Graph)
# MAGIC 
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC 
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC 
# MAGIC * __b) short response:__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it list of neighbors to be an empty set
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC 
# MAGIC 
# MAGIC * __d) code:__ Run the provided code to confirm that your job in `part a` has a record for each node and that your should records match the format specified in the docstring and the count should match what you computed in question 5. [__`TIP:`__ _you might want to take a moment to write out what the expected output should be fore the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC 
# MAGIC > __a)__ $N$ is the total number of nodes, i.e. number of pages in the graph that a random web-surfer would be traversing. In our case of Wiki page, N is 15,192,277. We initialize each node's rank to $$\frac{1}{N}$$ so that each page has an equal probability of being the first page in the "random surf"; it is a uniform probability distribution across all pages.
# MAGIC 
# MAGIC > __b)__ It will be more efficient to compute $N$ _after_ initializing records for each dangling node, otherwise $N$ would not be accurate as it would need to be updated to include the number of dangling nodes along with non-dangling nodes.

# COMMAND ----------

def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############    
    # write any helper functions here
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    def iterate_edges(k_v):
      """Loops through all nodes linked to a node and yields the node ID
      and an empty dict. Also yields the original graph structure."""
      key, value = k_v
      yield key, value
      for node_id, weight in value.items():
        yield node_id, dict()
    
    # write your main Spark code here    
    graphRDD = dataRDD.map(parse)\
                      .flatMap(iterate_edges)\
                      .reduceByKey(lambda k1, k2: {**k1, **k2})  # Combine all linked node dicts.
    N = graphRDD.count()
    graphRDD = graphRDD.map(lambda k_v: (k_v[0], (1/N, k_v[1])))  # Initialize all nodes as 1/N.
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

# part c - run your Spark job on the test graph (RUN THIS CELL AS IS)
start = time.time()
testGraph = initGraph(testRDD).collect()
print(f'... test graph initialized in {time.time() - start} seconds.')
testGraph

# COMMAND ----------

# part c - run your code on the main graph (RUN THIS CELL AS IS)
start = time.time()
wikiGraphRDD = initGraph(wikiRDD)
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part c - confirm record format and count (RUN THIS CELL AS IS)
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
print(f'First record: {wikiGraphRDD.take(1)}')
print(f'... initialization continued: {time.time() - start} seconds')

# COMMAND ----------

# MAGIC %md # Question 8 - PageRank part 2 (Iterate until convergence)
# MAGIC 
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC 
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The $P$ on the left hand side of the equation is the new score, and the $P$ on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, $|G|$ is the number of nodes in the graph (which we've elsewhere refered to as $N$).
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: $$\alpha * \frac{1}{|G|}$$
# MAGIC 
# MAGIC * __b) short response:__ In the equation for the PageRank calculation above what does $m$ represent and why do we divide it by $|G|$?
# MAGIC 
# MAGIC * __c) short response:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies teleportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC   * 
# MAGIC   
# MAGIC    __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC 
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC 
# MAGIC > __a)__ The random jump factor `α` is sometimes called the “teleportation” factor. The teleport operation is defined as where a surfer jumps from a node to any other node in the web graph. This may happen if an address is typed into the URL bar of his browser. The destination of a teleport operation is modeled as being chosen uniformly at random from all web pages. In other words, if `N` is the total number of nodes in the web graph, the teleport operation takes the surfer to each node with probability `1/N` (chances of landing at any particular page). The surfer would also teleport to his present position with probability `1/N`. Since there is an `1/|G|` chance of landing at any particular page, where |G| is the number of nodes in the web-graph, with probablity alpha `α` (a.k.a teleportationi factor), the entire probablity of random jump to a particular page is `α * 1/|G|`.
# MAGIC 
# MAGIC > __b)__ `m` is the missing PageRank mass lost at the dangling nodes, and `|G|` is the number of nodes in the entire graph. we divide `m`  by `|G|` to calculate share of the lost PageRank mass that is redistributed to each node in the graph.
# MAGIC 
# MAGIC > __c)__ For the iterative algorithm to work we need to have the exact the same data structure as the beginning, after each iteration.  PageRank itself is a probability distribution which is spreading probability mass to neighbors via outgoing links. Therefore PageRank values of all nodes or the total probabilty mass should sum up to 1 after each iteration.

# COMMAND ----------

# part d - provided FloatAccumulator class (RUN THIS CELL AS IS)

from pyspark.accumulators import AccumulatorParam

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We strongly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.
    
    # Calculate number of nodes for dividing teleportation factor:
    N = sc.broadcast(graphInitRDD.count())
    teleportation_mass = sc.broadcast(a.value / N.value)
    
    def map_non_dangling_mass(record):
      """Emit the weights for non-dangling nodes, along with 
      the original graph structure for the next iteration."""
      node_id, (mass, neighbors) = record
      edge_count = sum(neighbors.values())
      
      # Pass through the original graph structure with zero mass:
      yield node_id, (0, neighbors)
       
      # Iterate each edge with its share of the redistributed mass:
      if edge_count:
        mass_div_edge_count = mass / edge_count
        for n_id, weight in neighbors.items():
          neighbor_mass = mass_div_edge_count * weight
          yield n_id, (neighbor_mass, dict())

    def reducer(x, y):
      """Sum weights and combine dictionaries of edges."""
      x_mass, x_neighbors = x[0], x[1]
      y_mass, y_neighbors = y[0], y[1]
      total_mass = x_mass + y_mass
      all_neighbors = {**x_neighbors, **y_neighbors}
      return (total_mass, all_neighbors)
    
    def add_dangling_mass(record):
      """Add the constant dangling mass to all nodes."""
      node_id, (mass, neighbors) = record
      return node_id, (mass + dangling_mass_to_add.value, neighbors)
    
    def apply_teleportation(record):
      """Apply the 'teleportation' scaling factor alpha."""
      node_id, (mass, neighbors) = record
      mass *= d.value
      mass += teleportation_mass.value
      return node_id, (mass, neighbors)
            
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace
    if verbose:
      print("|  ".join([c.ljust(15) for c in ("Iter", "Total Mass", "Dangling Mass")]))

    prev_iter_rdd = graphInitRDD
    for iteration in range(1, maxIter+1, 1):
      
      # Calculate total dangling node mass:
      mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      prev_iter_rdd.filter(lambda x: not len(x[1][1])).foreach(lambda x: mmAccum.add(x[1][0]))
      dangling_mass_to_add = sc.broadcast(mmAccum.value / N.value)
            
      # Apply PageRank:
      steadyStateRDD = prev_iter_rdd.flatMap(map_non_dangling_mass)\
                                    .reduceByKey(reducer)\
                                    .map(add_dangling_mass)\
                                    .map(apply_teleportation)\
                                    .cache()
      
      # Confirm the total mass after each iteration still sums to 1:
      totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
      steadyStateRDD.foreach(lambda x: totAccum.add(x[1][0]))
      
      if verbose:
        print("|  ".join([i.ljust(15) for i in (f"{iteration}", f"{totAccum.value:.9f}", f"{mmAccum.value:.9f}")]))
      
      prev_iter_rdd = steadyStateRDD
    
    # Just yield the page ranks without the full graph:
    steadyStateRDD = steadyStateRDD.map(lambda k_v: (k_v[0], k_v[1][0]))
    
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

# COMMAND ----------

# part d - run PageRank on the test graph (RUN THIS CELL AS IS)
# NOTE: while developing your code you may want turn on the verbose option
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```

# COMMAND ----------

# part d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTE: wikiGraphRDD should have been computed & cached above!
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

top_20 = full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

top_20

# COMMAND ----------

# MAGIC %md __`expected results for the full graph:`__
# MAGIC ```
# MAGIC top_20 = 
# MAGIC [(13455888, 0.0015447247129832947),
# MAGIC  (4695850, 0.0006710240718906518),
# MAGIC  (5051368, 0.0005983856809747697),
# MAGIC  (1184351, 0.0005982073536467391),
# MAGIC  (2437837, 0.0004624928928940748),
# MAGIC  (6076759, 0.00045509400641448284),
# MAGIC  (4196067, 0.0004423778888372447),
# MAGIC  (13425865, 0.00044155351714348035),
# MAGIC  (6172466, 0.0004224002001845032),
# MAGIC  (1384888, 0.0004012895604073632),
# MAGIC  (6113490, 0.00039578924771805474),
# MAGIC  (14112583, 0.0003943847283754762),
# MAGIC  (7902219, 0.000370098784735699),
# MAGIC  (10390714, 0.0003650264964328283),
# MAGIC  (12836211, 0.0003619948863114985),
# MAGIC  (6237129, 0.0003519555847625285),
# MAGIC  (6416278, 0.00034866235645266493),
# MAGIC  (13432150, 0.00033936510637418247),
# MAGIC  (1516699, 0.00033297500286244265),
# MAGIC  (7990491, 0.00030760906265869104)]
# MAGIC ```

# COMMAND ----------

# Save the top_20 results to disc for use later. So you don't have to rerun everything if you restart the cluster.

# Created a directory under dbfs
dbutils.fs.mkdirs("/team8_temp")

# Convert to dataframe
top_20_df = pd.DataFrame(top_20)

# Save to dbfs temp folder
top_20_df.to_csv('/dbfs/team8_temp/team8_top_20.csv')

# COMMAND ----------

temp_path = "/team8_temp"
# read Top_20 results csv from dbfs
top20 = spark.read.csv((f"{temp_path}"))
display(top20)

# COMMAND ----------

# view record from indexRDD (RUN THIS CELL AS IS)
# title\t indx\t inDeg\t outDeg
indexRDD.take(1)

# COMMAND ----------

# map indexRDD to new format (index, name) (RUN THIS CELL AS IS)
namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

# see new format (RUN THIS CELL AS IS)
namesKV_RDD.take(2)

# COMMAND ----------

# MAGIC %md # OPTIONAL
# MAGIC ### The rest of this notebook is optional and doesn't count toward your grade.
# MAGIC The indexRDD we created earlier from the indices.txt file contains the titles of the pages and thier IDs.
# MAGIC 
# MAGIC * __a) code:__ Join this dataset with your top 20 results.
# MAGIC * __b) code:__ Print the results

# COMMAND ----------

# MAGIC %md ## Join with indexRDD and print pretty

# COMMAND ----------

# part a
joinedWithNames = None
############## YOUR CODE HERE ###############

############## END YOUR CODE ###############

# COMMAND ----------

# part b
# Feel free to modify this cell to suit your implementation, but please keep the formatting and sort order.
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in joinedWithNames:
    print ("{:6f}\t| {:10d}\t| {}".format(r[1][1],r[0],r[1][0]))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## OPTIONAL - GraphFrames
# MAGIC GraphFrames is a graph library which is built on top of the Spark DataFrames API.
# MAGIC 
# MAGIC * __a) code:__ Using the same dataset, run the graphframes implementation of pagerank.
# MAGIC * __b) code:__ Join the top 20 results with indices.txt and display in the same format as above.
# MAGIC * __c) short answer:__ Compare your results with the results from graphframes.
# MAGIC 
# MAGIC __NOTE:__ Feel free to create as many code cells as you need. Code should be clear and concise - do not include your scratch work. Comment your code if it's not self annotating.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC ### You will need to generate vertices (v) and edges (e) to feed into the graph below. 
# MAGIC Use as many cells as you need for this task.

# COMMAND ----------

# Create a GraphFrame
from graphframes import *
g = GraphFrame(v, e)


# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

start = time.time()
top_20 = results.vertices.orderBy(F.desc("pagerank")).limit(20)
print(f'... completed job in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %%time
# MAGIC top_20.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Run the cells below to join the results of the graphframes pagerank algorithm with the names of the nodes.

# COMMAND ----------

namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

namesKV_DF = namesKV_RDD.toDF()

# COMMAND ----------

namesKV_DF = namesKV_DF.withColumnRenamed('_1','id')
namesKV_DF = namesKV_DF.withColumnRenamed('_2','title')
namesKV_DF.take(1)

# COMMAND ----------

resultsWithNames = namesKV_DF.join(top_20, namesKV_DF.id==top_20.id).orderBy(F.desc("pagerank")).collect()

# COMMAND ----------

# TODO: use f' for string formatting
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in resultsWithNames:
    print ("{:6f}\t| {:10s}\t| {}".format(r[3],r[2],r[1]))

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW5! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform
