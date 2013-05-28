#pragma once
#include "graph.hpp"
#include "NetFunDll.h"
namespace scn
{
/**
 * @name GenKNearestNetwork - Generates a K-Nearest network
 * @param numberOfNodes -  number Of Nodes 
 * @param k -  each node will connect with forward k nodes and
 * backward k nodes. the degree of each node is 2*k 
 * @return UGraph::pGraph return the generated network
 */
   NETWORKLIBDLL UGraph::pGraph GenKNearestNetwork(size_t numberOfNodes, size_t k);
/**
 * @name GenSmallWorldNetworkByWS - Generates WS small world network
 * @param numberOfNodes - Number of nodes
 * @param k - same with GenKNearestNetwork
 * @param probability - probability of re-connection 
 * @return UGraph::pGraph - the network
 */
   NETWORKLIBDLL UGraph::pGraph GenSmallWorldNetworkByWS(size_t numberOfNodes, size_t k,double probability);
/**
 * @name GenSmallWorldNetworkByNW - Generates a NW small world network
 * @param numberOfNodes - Number of Nodes
 * @param k - same with GenKNearestNetwork and GenSmallWolrdNetworkByWS
 * @param probability - probability of additive connection
 * @return UGraph::pUGraph
 */
   NETWORKLIBDLL UGraph::pGraph GenSmallWorldNetworkByNW(size_t numberOfNodes,size_t k,double probability);
  
   /**
 * @name GenTreeStructuredSW - Generates small-world
 * network based truee structure
 * @param times -  number Of Iterations 
 * @return UGraph::pGraph
 */
   NETWORKLIBDLL UGraph::pGraph GenTreeStructuredSW(size_t times);  
 /**
 * @name GenTreeStructuredSFSW - Generates a scale-free small-world
 * network based truee structure
 * @param numberOfNodes -  number Of Nodes 
 * @return UGraph::pGraph
 */
   //Proposed by Zhe-Ming Lu, Yu-Xin Su and Shi-Ze Guo, published in Physica A, Vol. xx, No. xx, 201x
  //Title: Deterministic scale-free small-world network of arbitrary order 
   NETWORKLIBDLL UGraph::pGraph GenTreeStructuredSFSW(size_t numberOfNodes);


   NETWORKLIBDLL UGraph::pGraph GenRelationNetwork(size_t numberOfNodes);

   NETWORKLIBDLL UGraph::pGraph GenPreferenceMemoryNetwork(size_t numberOfNodes);

   double PearsonCoefficient(vector<double> x, vector<double> y);



   void GetWccs(UGraph::pGraph& graph, vector< vector<size_t>>& CnComV);
   void CmtyGirvanNewmanStep(UGraph::pGraph& graph, vector<size_t>& Cmty1, vector<size_t>& Cmty2);
   void GetBetweennessCentralityOfEdge(UGraph::pGraph& graph, vector< pair<pair<size_t, size_t>, double>> BtwEH);
   pair<size_t, size_t> GetNumberOfShortestPathEdge(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget, size_t indexOfThrough_A, size_t indexOfThrough_B);
   void GetNodeWcc(UGraph::pGraph& graph, const size_t& NId, vector<size_t>& CnCom);
   double GirvanNewmanGetModularity(UGraph::pGraph& G, const std::unordered_map<size_t, size_t> OutDegH, const int& OrigEdges, vector< vector<size_t>>& CnComV);
   NETWORKLIBDLL double CommunityGirvanNewman(UGraph::pGraph& graph, vector< vector<size_t>>& CmtyV);


/**
 * @name GenRandomNetwork - Generates ER random network
 * @param numberOfNodes - Number of number Of Nodes 
 * @param probability - probability of connection 
 * @return UGraph::pGraph
 */
   NETWORKLIBDLL UGraph::pGraph GenRandomNetwork(size_t numberOfNodes,double probability);
/**
 * @name GenScaleFreeNetwork - Generate scale free or BA network
 * @param numberOfNodes - Number of number Of Nodes 
 * @param degree - Number of degree of new node
 * @return UGraph::pGraph
 */
   NETWORKLIBDLL UGraph::pGraph GenScaleFreeNetwork(size_t numberOfNodes, size_t degree);
/**
 * @name GenSmallWorldByEdgeIteration - Generates a deterministic
 * small-world network by edge iteration.
 * @param times -  times for iteration
 * @return UGraph::pGraph
 * @ref 2006_Z.Zhang_A deterministic small-world network created by edge iterations
 */
   NETWORKLIBDLL UGraph::pGraph GenSmallWorldByEdgeIteration(size_t times);
/**
 * @name GenUniformRecursiveTree - Generates a Uniform Recursive Tree(URT)
 * @param numberOfNodes - number of nodes
 * @return UGraph::pGraph
 * @ref 1995_R.T.Smythe_A survey of recursive trees
 */
   NETWORKLIBDLL UGraph::pGraph GenUniformRecursiveTree(size_t numberOfNodes);
/**
 * @name GenDURT - Generates a Deterministic Uniform Recursive Tree 
 * @param times -  times of iteration
 * @return UGraph::pGraph
 * @ref 2008_Z.Zhang_Topologies and Laplacian spectra of a
 * deterministic uniform recursive tree
 */
   NETWORKLIBDLL UGraph::pGraph GenDURT(size_t times);
/**
 * @name GenSmallWorldNetworkFromDURT - Generates a deterministic
 * small-world network from a Deterministic Uniform Recursive Tree
 * @param times -  times of iteration
 * @return UGraph::pGraph
 * @ref 2012_Z.M.Lu_A small-world network derived from the
 * deterministic uniform recursive tree
 */
   NETWORKLIBDLL UGraph::pGraph GenSmallWorldNetworkFromDURT(size_t times);
/**
 * @name GenCommunityNetwork - Generate the simplest community network
 * @param numberOfNodes -  number Of Nodes 
 * @param numberOfCommunity -  number Of Community 
 * @param inner_prob - probability for the linking within a single community
 * @param outer_prob - probability for the linking between two communities 
 * @return UGraph::pGraph
 */
   NETWORKLIBDLL UGraph::pGraph GenCommunityNetwork(size_t numberOfNodes, size_t numberOfCommunity,double inner_prob, double outer_prob);

/**
 * @name GenCommunityNetwork - Generate the DSW network based on Triaganle extension
 * @param iteration_times -  times of iteration 
 * @return UGraph::pGraph
 */ 
   ////Proposed by Shi-Ze Guo, Xin-Feng Li,Zhe-Ming Lu, and Zhe Chen, published in ADS, Vol. 65, No. 12, 2013
   //Title: A Triangle-Extended Deterministic Small-World Network
   NETWORKLIBDLL UGraph::pGraph GenTriangleExtendedDSWN(size_t iteration_times);
   NETWORKLIBDLL UGraph::pGraph GenSwirlShapedNetwork(size_t times);
   NETWORKLIBDLL UGraph::pGraph GenPinWheelShapedSW(size_t times);

/**
 * @name GenNetworkFromDegreeDistribution - Generates a new string of
 * type p UGraph List 
 * @param degree_list - degree distribution. the first is number of
 * degree, the second is number of nodes
 * @param list - a list of graphs which fit the same 
 * degree distribution (return)
 */
   //unfinished algorithm
   void GenNetworkFromDegreeDistribution(pUGraphList &list,std::unordered_map<size_t,size_t> &degree_list);
/**
 * @name IsDegreeListGraphical - test whether a given degree
 * distribution can construct a graph. This method is based on
 * Havel-Hakimi algorithm
 * @param degree_distribution - degree distribution 
 * @return bool - return true if this degre distribution can contruct
 * a connected graph.
 */
   NETWORKLIBDLL bool IsDegreeListGraphical(std::unordered_map<size_t,size_t> &degree_distribution);
/**
 * @name RenormalizeByBoxCounting - a given undirected network is
 * renormalized by the method of box-counting described in Ref. This
 * method belongs to network clutering
 * @param graph -  graph, the given undirected network
 * @param length -  the diameter of all new box is less than this "length"
 * @return UGraph::pGraph - the renormalized undirected network
 * @ref 2005_Nature_C.Song_Self-similarity of complex networks
 */
   NETWORKLIBDLL UGraph::pGraph RenormalizeByBoxCounting(UGraph::pGraph graph, size_t length);


   NETWORKLIBDLL double RenormalizeBySpectralBisection(UGraph::pGraph graph, size_t length);




 /**
 * @name WriteToNetFile - write a given undirected network to a .net file
 * @param path -  the given file path
 * @param network -  the given network
 */
   NETWORKLIBDLL void WriteToNetFile(char * path, UNetwork<>::pNetwork &network);

/**
 * @name ReadUNetworkFromNetFile - read an undirected network from a present .net file 
 * @param path -  the given file path
 * @param network -  the given network
 */
   NETWORKLIBDLL void ReadUNetworkFromNetFile(UNetwork<>::pNetwork &network,char * path);

 /**
 * @name WriteToDotFile - write the structure of an undirected
 * graph into the dot form, which used as input of GraphViz
 * @param graph -  graph 
 * @param filename - Name of
 * @return void
 */
   NETWORKLIBDLL void WriteToDotFile(Graph::pGraph graph, std::string filename);
/**
 * @name DrawCircleForm - write the structure of an undirected
 * graph into dot form, then call GraphViz to render this graph
 * @param graph -  graph 
 * @param filename - the name of dot file and rendered image file
 * @return void
 */
   NETWORKLIBDLL void DrawCircleForm(Graph::pGraph graph,std::string filename);
};
