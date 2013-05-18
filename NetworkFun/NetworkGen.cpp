// NetworkGen.cpp : 定义NetworkFun V2.0关于网络生成部分DLL应用程序的导出函数。
//
#include "stdafx.h"
#include "NetworkGen.hpp"
#include "Feature&search.hpp"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <list>
#include <stack>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace scn;

void scn::WriteToNetFile(char* path, UNetwork<>::pNetwork &network)
   {
      using std::endl;
	  std::ofstream outfile(path,ios_base::trunc);
      UGraph::pGraph graph = network->GetTopology();
      
      outfile<<"*Vertices "<<graph->GetNumberOfNodes()<<endl;
      //write node
      for(auto node = graph->begin(); node != graph->end(); node++)
      {
	 auto position = network->GetNodePosition(*node);
	 outfile<<*node + 1<<" "<<*node + 1<<" "<<position[0]<<" "
		<<position[1]<<" "<<position[2]<<endl;
      }
      
      outfile<<"*Arcs"<<endl;
      outfile<<"*Edges"<<endl;

      //write edge
      for(auto node = graph->begin(); node != graph->end(); node++)
      {
	 for(auto other = node->begin(); other != node->end(); other++)
	 {
	    if(*other < *node)
	    {
	       outfile<<*node + 1<<" "<<*other + 1<<" 1"<<endl;
	    }
	 }
      }
      outfile.close();
   }

  void scn::ReadUNetworkFromNetFile(UNetwork<>::pNetwork &network,char * path)
   {
      using std::getline;
      using std::string;
      using std::cout;
      using std::endl;
      std::ifstream infile(path);
      UGraph::pGraph graph(new UGraph());
      network.reset(new UNetwork<>(graph));
	  
      string line;
      string temp;
      std::stringstream ss;
      //read header
      while(getline(infile, line))
      {
	 ss.str(line);
	 if(ss>>temp && temp == "*Vertices")
	 {//read nodes
	    size_t numberOfNodes;
	    ss>>numberOfNodes;
	    size_t index;
	    string flag;
	    double x,y,z;
	    for(size_t i = 0; i < numberOfNodes; i++)
	    {
	       getline(infile, line);
	       ss.clear();
	       ss.str(line);
	       if(ss>>index>>flag>>x>>y>>z)//read content
	       {
		  graph->AddNode(index - 1);
		  network->SetNodePosition(index - 1, x, y, z);
	       }
	    }
	    assert(numberOfNodes == graph->GetNumberOfNodes());
	 }
	 else if(line == "*Edges")
	 {//read edge
	    size_t indexOfNode1, indexOfNode2;
	    double weight;
	    while(getline(infile, line))
	    {
	       ss.clear();
	       ss.str(line);
	       if(ss>>indexOfNode1>>indexOfNode2>>weight)
		  graph->AddEdge(indexOfNode1 - 1, indexOfNode2 -1);
	    }
	 }
      }
      infile.close();

   }

void scn::DrawCircleForm(Graph::pGraph graph, std::string filename)
{
   WriteToDotFile(graph, filename);

   std::string temp("circo -Tpng ");
   temp += filename + ".dot";
   temp += " -o " + filename +".png";
   
   if(std::system(temp.c_str()) != 0)
   {
      assert(false);
   }
}

void scn::WriteToDotFile(Graph::pGraph graph, std::string filename)
{
   ofstream out_file(filename + ".dot", ios_base::out | ios_base::trunc);
   //print the head
   out_file<<"graph G {"<<endl;
   //print edges list
   for(auto node = graph->begin(); node != graph->end(); node++)
   {
      size_t node_id = node->GetIndexOfNode();
      for(auto other = node->begin(); other != node->end(); other++)
      {
	 if(node_id > *other && graph->HasEdge(node_id, *other))
	    continue;
	 
	 out_file<<"   "<<node_id<<" -- "<<*other<<endl;
      }
   }
   //print the tail
   out_file<<"}"<<endl;
   out_file.close();
}
UGraph::pGraph scn::GenKNearestNetwork(size_t numberOfNodes, size_t k)
{
   assert(2*k + 1 <= numberOfNodes);

   UGraph::pGraph graph(new UGraph(numberOfNodes));
   
  //generate edges
   for(auto node = graph->begin(); node != graph->end(); node++)
   {
      for(size_t j = 1; j <= k; j++)
      {
	 //forward
	 graph->AddEdge(node, (*node + j) % numberOfNodes);
      }
   }
   return graph;
}

UGraph::pGraph scn::GenCommunityNetwork(size_t numberOfNodes, size_t numberOfCommunity,
					double inner_prob, double outer_prob)
{
   assert(numberOfCommunity < numberOfNodes);
   assert((1 - inner_prob) * inner_prob >= 0);// 0 <= inner_prob <= 1
   assert((1 - outer_prob) * outer_prob >= 0);// 0 <= outer_prob <= 1
   //
   vector<vector<size_t>> communities(numberOfCommunity);
   size_t community_size = numberOfNodes / numberOfCommunity;
   for(size_t i = 0; i < numberOfNodes; i++)
   {
      if(i / community_size >= communities.size())
	 communities.rbegin()->push_back(i);
      else
	 communities[i / community_size].push_back(i);
   }
   //create network
   UGraph::pGraph graph(new UGraph(numberOfNodes));
   
//inner link
   srand(size_t(time(00)));
   for(int ii=0;ii<communities.size();ii++)
   {
      size_t numberOfEdge = static_cast<size_t>(
	 communities[ii].size() * (communities[ii].size() - 1) / 2 * inner_prob);
      size_t current_edge = 0;
      size_t one,two;
      while(current_edge < numberOfEdge)
      {
	 one = rand() % communities[ii].size();
	 two = rand() % communities[ii].size();
	 if(one == two || graph->HasEdge(communities[ii][one], communities[ii][two]))
	    continue;
	 //else
	 current_edge++;
	 graph->AddEdge(communities[ii][one], communities[ii][two]);
      }
   }
//outer link
   for(size_t i = 0; i < numberOfCommunity - 1; i++)
   {
      auto& comm_one = communities[i];
      for(size_t j = i + 1; j < numberOfCommunity; j++)
      {
	 auto& comm_two = communities[j];
	 size_t numberOfEdge = static_cast<size_t>(
	    comm_one.size() * comm_two.size() * outer_prob / 2);
	 size_t current_edge = 0, one, two;
	 while(current_edge < numberOfEdge)
	 {
	    one = rand() % comm_one.size();
	    two = rand() % comm_two.size();

	    if(!graph->HasEdge(comm_one[one], comm_two[two]))
	    {
	       current_edge++;
	       graph->AddEdge(comm_one[one], comm_two[two]);
	    }
	 }
      }
   }
   return graph;
}

UGraph::pGraph scn::GenSmallWorldNetworkByWS(size_t numberOfNodes, size_t k,
						   double probability)
{
   assert(2*k + 1 <= numberOfNodes);
   assert(probability >= 0);
   assert(probability <= 1);
   //generate node
   UGraph::pGraph graph(new UGraph(numberOfNodes));
   
   //generate edge
   srand(size_t(time(00)));

   for(auto node = graph->begin(); node != graph->end(); node++)
   {
      size_t other;
      for(size_t j = 1; j <= k; j++)
      {
	 other = (*node + j) % numberOfNodes;

	 //random selection
	 if(double(rand() % 65536) / 65536 < probability)
	 {
	    do
	    {
	       other = rand() % numberOfNodes;
	    }while(other == *node || graph->HasEdge(node, other));
	 }
	 //add
	 graph->AddEdge(node, other);
      }
   }
   return graph;
}

UGraph::pGraph scn::GenSmallWorldNetworkByNW(size_t numberOfNodes, size_t k,
						   double probability)
{
   assert(2* k + 1 <= numberOfNodes);
   assert(probability >= 0);
   assert(probability <= 1);
   
   //generate k-nearest network
   UGraph::pGraph graph = scn::GenKNearestNetwork(numberOfNodes, k);
   //add edges randomly
   size_t numberOfEdges = numberOfNodes * (numberOfNodes - 2 * k - 1) / 2 * probability;
   size_t sum_edges = 0;
   size_t one, two;
   srand(size_t(time(00)));

   do
   {
      one = rand() % numberOfNodes;
      two = rand() % numberOfNodes;
      if(!(one == two || graph->HasEdge(one, two)))
      {
	 graph->AddEdge(one, two);
	 sum_edges++;
      }
   }while(sum_edges < numberOfEdges);

   return graph;
}

UGraph::pGraph scn::GenRandomNetwork(size_t numberOfNodes, double probability)
{
   assert(probability >= 0);
   assert(probability <= 1);

   UGraph::pGraph graph(new UGraph(numberOfNodes));

   size_t numberOfEdges = numberOfNodes * (numberOfNodes - 1) / 2 * probability;
   size_t sum_edges = 0;
   size_t one, two;
   //generate edges
   srand(size_t(time(00)));

   do
   {
      one = rand() % numberOfNodes;
      two = rand() % numberOfNodes;
      if(!(one == two || graph->HasEdge(one, two)))
      {
	 graph->AddEdge(one, two);
	 sum_edges++;
      }
   }while(sum_edges < numberOfEdges);

   return graph;
}

UGraph::pGraph scn::GenScaleFreeNetwork(size_t numberOfNodes, size_t degree)
{
   assert(degree < numberOfNodes);
   //init
   srand(size_t(time(00)));
   UGraph::pGraph graph(new UGraph());
   std::vector<size_t> node_history;//it records which nodes are to be
				    //connected in each times. The
				    //number of every index in node
				    //history represents the degree of
				    //the node. The more times the
				    //index appears, the high
				    //probability it will be chosen to connect
   node_history.reserve(2 * numberOfNodes * degree);
   //add first two nodes
   graph->AddNode(0);
   graph->AddNode(1);
   graph->AddEdge(0, 1);
   node_history.push_back(0);
   node_history.push_back(1);
   std::unordered_set<size_t> node_set;
//   node_set.reserve(degree);
   //add
   for(size_t i = 2; i < numberOfNodes; i++)
   {
      node_set.clear();
      while(node_set.size() < degree && node_set.size() < i)
      {
	 node_set.insert(node_history[ rand() % node_history.size() ]);
      }
      graph->AddNode(i);
      // auto iter = graph->find(i);
      for(auto other = node_set.begin(); other != node_set.end(); other++)
      {
	 graph->AddEdge(i, *other);
	 node_history.push_back(*other);
	 node_history.push_back(i);
      }
   }
   return graph;
}

UGraph::pGraph scn::GenSmallWorldByEdgeIteration(size_t times)
{
   //init
   UGraph::pGraph graph(new UGraph());
   list<pair<size_t, size_t>> new_edges;
   list<pair<size_t, size_t>> previous_new_edges;
   //time 0
   for(size_t i = 0; i < 3; i++)

   {
      graph->AddNode(i);
   }
   for(size_t i = 0; i < 3; i++)
   {
      graph->AddEdge(i, (i+1) % 3);
      new_edges.push_back(make_pair(i, (i+1) % 3));
   }
   //add
   for(size_t i = 0; i < times; i++)
   {
      previous_new_edges.clear();
      swap(previous_new_edges, new_edges);
      for(auto edge = previous_new_edges.begin(); 
	  edge != previous_new_edges.end(); edge++)
      {
	 size_t new_node = graph->AddNode();
	 //triple(new_node, edge->first, edge->second)
	 graph->AddEdge(new_node, edge->first);
	 graph->AddEdge(new_node, edge->second);
	 new_edges.push_back(make_pair(new_node, edge->first));
	 new_edges.push_back(make_pair(new_node, edge->second));
      }
   }
   return graph;
}

UGraph::pGraph scn::GenUniformRecursiveTree(size_t numberOfNodes)
{
   assert(numberOfNodes >= 2);

   //init
   UGraph::pGraph graph(new UGraph());
   graph->AddNode(0);
   graph->AddNode(1);
   graph->AddEdge(0, 1);
   srand(size_t(time(00)));
   size_t target, new_node;
   //add
   for(size_t i = 2; i < numberOfNodes; i++)
   {
      target = rand() % graph->GetNumberOfNodes();
      new_node = graph->AddNode();
      graph->AddEdge(target, new_node);
   }
   return graph;
}

UGraph::pGraph scn::GenDURT(size_t times)
{
   //init
   UGraph::pGraph graph(new UGraph());
   graph->AddNode(0);
   graph->AddNode(1);
   graph->AddEdge(0, 1);
   size_t old_size;
   //add
   for(size_t i = 0; i < times; i++)
   {
      old_size = graph->GetNumberOfNodes();
      for(size_t j = 0; j < old_size; j++)
      {
	 graph->AddEdge(j, graph->AddNode());
      }
   }
   return graph;
}

UGraph::pGraph scn::GenSmallWorldNetworkFromDURT(size_t times)
{
   //init
   UGraph::pGraph graph(new UGraph());
   graph->AddNode(0);
   graph->AddNode(1);
   graph->AddEdge(0, 1);
   size_t old_size, new_node;
   //add
   for(size_t i = 0; i < times; i++)
   {
      old_size = graph->GetNumberOfNodes();
      for(size_t j = 0; j < old_size; j++)
      {
	 new_node = graph->AddNode();
	 graph->AddEdge(new_node, j);//standard step
	 graph->AddEdge(new_node, old_size - j - 1);//extra step
      }
   }
   return graph;
}
//Proposed by Shi-Ze Guo, Xin-Feng Li,Zhe-Ming Lu, and Zhe Chen, published in ADS, Vol. 65, No. 12, 2013
//Title: A Triangle-Extended Deterministic Small-World Network
UGraph::pGraph scn::GenTriangleExtendedDSWN(size_t iteration_times)
{
    //init
    UGraph::pGraph graph(new UGraph());
    list<size_t> new_nodes;
    list<size_t> previous_new_nodes;

    //time 0: a simple triangle
    for(size_t i = 0;i < 3;i++)
    {
        graph->AddNode(i);
        new_nodes.push_back(i);
    }
    for(size_t i = 0;i < 3;i++)
    {
        graph->AddEdge(i,(i +1) % 3);
    }

    //add
    for(size_t i = 0;i < iteration_times;i++)
    {
        previous_new_nodes.clear();
        swap(previous_new_nodes, new_nodes);
        for(auto node = previous_new_nodes.begin();
            node != previous_new_nodes.end();node++)
        {
            size_t leftnode_index = (*node)*2 + 3;
            size_t rightnode_index = leftnode_index + 1;

            //add triangle between(*node, leftnode_index, rightnode_index)
            graph->AddNode(leftnode_index);
            graph->AddNode(rightnode_index);
            graph->AddEdge(*node,leftnode_index);
            graph->AddEdge(*node,rightnode_index);
            graph->AddEdge(leftnode_index,rightnode_index);

            new_nodes.push_back(leftnode_index);
            new_nodes.push_back(rightnode_index);
        }
    }
    return graph;
}




UGraph::pGraph scn::GenSwirlShapedNetwork(size_t times)
{
   UGraph::pGraph graph(new UGraph(4));
   //create the first complete graph
   graph->AddEdge(0, 1);
   graph->AddEdge(0, 2);
   graph->AddEdge(0, 3);
   graph->AddEdge(1, 2);
   graph->AddEdge(1, 3);
   graph->AddEdge(2, 3);
   //normal pr0cedure
   array<size_t, 4> new_nodes;
   while(graph->GetNumberOfNodes() < 4 * (times + 1))
   {
      //add new nodes
      new_nodes[0] = graph->AddNode();
      new_nodes[1] = graph->AddNode();
      new_nodes[2] = graph->AddNode();
      new_nodes[3] = graph->AddNode();
      //outer circle
      graph->AddEdge(new_nodes[0], new_nodes[1]);
      graph->AddEdge(new_nodes[1], new_nodes[2]);
      graph->AddEdge(new_nodes[2], new_nodes[3]);
      graph->AddEdge(new_nodes[3], new_nodes[0]);
      //inter and outer
      for(size_t i = 0; i < 4; i++)
      {
	 graph->AddEdge(new_nodes[i], new_nodes[i] - 4);
	 if(new_nodes[i] != 4)
	 {
	    graph->AddEdge(new_nodes[i], new_nodes[i] - 5);
	 }
	 else
	 {
	    graph->AddEdge(4, 3);
	 }
      }
   }
   return graph;
}

UGraph::pGraph scn::GenPinWheelShapedSW(size_t times)
{
   UGraph::pGraph graph(new UGraph(5));
   //first complete graph with 5 nodes
   for(size_t i = 0; i < 5; i++)
   {
      for(size_t j = i + 1; j < 5; j++)
      {
	 graph->AddEdge(i, j);
      }
   }
   //normal procedure
   array<size_t, 4> new_nodes;
   size_t current_times = 0;
   while(current_times < times)
   {
      //add new nodes
      new_nodes[0] = graph->AddNode();
      new_nodes[1] = graph->AddNode();
      new_nodes[2] = graph->AddNode();
      new_nodes[3] = graph->AddNode();
      //add edge
      graph->AddEdge(new_nodes[current_times++ % 4], 0);
      for(size_t i = 0; i < 4; i++)
      {
	 graph->AddEdge(new_nodes[i], new_nodes[(i + 1) % 4]);
	 graph->AddEdge(new_nodes[i], new_nodes[i] - 4);
	 if(new_nodes[i] != 5)
	 {
	    graph->AddEdge(new_nodes[i], new_nodes[i] - 5);
	 }
	 else
	 {
	    graph->AddEdge(5, 4);
	 }
      }
   }
   return graph;
}
//unfinished algorithm
void scn::GenNetworkFromDegreeDistribution(pUGraphList &list,std::unordered_map<size_t,size_t> &degree_list)
{
   size_t numberOfNodes = 0;
   size_t numberOfEdges = 0;
   vector<size_t> max_degree;//max degree of each node
   for(auto degree = degree_list.begin();
       degree != degree_list.end(); degree++)
   {
      numberOfNodes += degree->second;
      numberOfEdges += degree->first * degree->second;
      max_degree.insert(max_degree.end(), degree->second, degree->first);
   }
   numberOfEdges /= 2;
   //try
   UGraph::pGraph base_graph(new UGraph(numberOfNodes));
   stack<pair<size_t, size_t>> edge_recorder;//record the process of
					     //adding edges
   //edge_recorder.reserve(numberOfEdges);
   
   //return pUGraphList();
}

bool scn::IsDegreeListGraphical(std::unordered_map<size_t,size_t> &degree_distribution)
{
   vector<int> degree_list;
   for(auto iter = degree_distribution.begin();
       iter != degree_distribution.end(); iter++)
   {
      degree_list.insert(degree_list.end(), iter->second, iter->first);
   }
   //test
   while(!degree_list.empty())
   {
      sort(degree_list.begin(), degree_list.end());
      int max = *degree_list.rbegin();
      auto iter = degree_list.rbegin() + 1;//other nodes linked with rbegin()
      for(int i = 0; i < max; i++)
      {
	 if(iter == degree_list.rend())
	 {
	    return false;//if ther is no enough lower degree node,
			 //this sequence is not graphcial
	 }
	 else if(*iter == 0)
	 {
	    return false;
	 }
	 else
	 {
	    (*iter++)--;//linked with the highest degree, the number
			//of extra degree decreases by 1
	 }
      }
      degree_list.pop_back();
   }
   return true;
}


UGraph::pGraph scn::RenormalizeByBoxCounting(UGraph::pGraph graph, size_t length)
{
   UGraph::pGraph temp_graph(new UGraph(*graph));
   std::unordered_map<size_t,size_t> group_index;//stores the index of group of each
			   //previous nodes. pairs(index in graph,
			   //index of group)
   size_t numberOfGroups = 0;
   
   while(!temp_graph->empty())
   {
      auto list =FindClosureGroup(temp_graph,*(temp_graph->begin()), length);
      
      for(auto i=list.begin();i!=list.end();i++)
      {
	    //add to group_index
	    group_index[*i] = numberOfGroups;
	    //remove from temp_graph
	    temp_graph->RemoveNode(*i);
      }
      numberOfGroups++;
   }
   //renormalize
   temp_graph.reset(new UGraph(numberOfGroups));
   //add edge
   for(auto i = group_index.begin(); i != group_index.end();i++)
   {
      for(auto j = i; j != group_index.end(); j++)
      {
	 if(i->second != j->second && graph->HasEdge(i->first, j->first))
	 {
	    temp_graph->AddEdge(i->second, j->second);
	 }
      }
   } 
   return temp_graph;
}













UGraph::pGraph scn::GenTreeStructuredSW(size_t times)
{
   UGraph::pGraph graph(new UGraph());
   //times == 0
   graph->AddNode(0);
   if(times == 0)
      return graph;
   //times == 1
   graph->AddNode(1);
   graph->AddNode(2);
   graph->AddEdge(0,1);
   graph->AddEdge(0,2);
   graph->AddEdge(1,2);
   if(times == 1)
      return graph;
   //begin to normal procedure
   queue<pair<size_t, vector<size_t>> > leaf_queue;//pair<self, ancestor>
   leaf_queue.push(make_pair(1, vector<size_t>(1)));
   leaf_queue.push(make_pair(2, vector<size_t>(1)));
   //construct node in breadth-first order
   size_t numberOfNodes = 1<<(times + 1) - 1;
   while(graph->GetNumberOfNodes() < numberOfNodes)
   {
      auto current = leaf_queue.front();
      leaf_queue.pop();
      //left node
      size_t left = graph->AddNode();
      graph->AddEdge(current.first, left);
      //add ancestor
      graph->AddEdge(left, current.second[current.second.size() - 1]);
      //right node
      size_t right = graph->AddNode();
      graph->AddEdge(current.first, right);
      graph->AddEdge(left, right);
      //add ancestor
      graph->AddEdge(right, current.second[current.second.size() - 1]);
      //add ancestor list
      vector<size_t> temp(current.second.begin(), current.second.end());
      temp.push_back(current.first);
      leaf_queue.push(make_pair(left, temp));
      leaf_queue.push(make_pair(right, temp));
   }
   return graph;
}
//Proposed by Zhe-Ming Lu, Yu-Xin Su and Shi-Ze Guo, published in Physica A, Vol. xx, No. xx, 201x
//Title: Deterministic scale-free small-world network of arbitrary order 
UGraph::pGraph scn::GenTreeStructuredSFSW(size_t numberOfNodes)
{
   UGraph::pGraph graph(new UGraph());
   //times == 0
   graph->AddNode(0);
   if(numberOfNodes == 1)
      return graph;
   //times == 1
   graph->AddNode(1);
   graph->AddNode(2);
   graph->AddEdge(0,1);
   graph->AddEdge(0,2);
   graph->AddEdge(1,2);
   if(numberOfNodes == 3)
      return graph;
   //begin to normal procedure
   queue<pair<size_t, vector<size_t>> > leaf_queue;//pair<self, ancestor>
   leaf_queue.push(make_pair(1, vector<size_t>(1)));
   leaf_queue.push(make_pair(2, vector<size_t>(1)));
   //construct node in breadth-first order
   size_t records = 0;
   while(graph->GetNumberOfNodes() < numberOfNodes)
   {
      auto current = leaf_queue.front();
      leaf_queue.pop();
      //left node
      size_t left = graph->AddNode();
      graph->AddEdge(current.first, left);
      //add ancestor
      graph->AddEdge(left, current.second[records % current.second.size()]);
      //right node
      size_t right = 0;
      if(graph->GetNumberOfNodes() != numberOfNodes)
      {
	 right = graph->AddNode();
	 graph->AddEdge(current.first, right);
	 graph->AddEdge(left, right);
	 //add ancestor
	 graph->AddEdge(right, current.second[records++ % current.second.size()]);
      }
      if(current.second.size() != leaf_queue.front().second.size())
      {//the next node is in a new level
	 records = 0;
      }
      //add ancestor list
      vector<size_t> temp(current.second.begin(), current.second.end());
      temp.push_back(current.first);
      leaf_queue.push(make_pair(left, temp));
      leaf_queue.push(make_pair(right, temp));
   }
   return graph;
}





/************************************************************************
* Paper: A Novel Social Network Model for Forming Relationships* 
* Author: 李辰
* Date: 20100319
************************************************************************/

UGraph::pGraph scn::GenRelationNetwork(size_t numberOfNodes)
{
	//adjustment of probability
	double probability = 0.07;
	
	//generate node	
	UGraph::pGraph graph = scn::GenCommunityNetwork(5, 2, 0.65, 0.55);
	   
	//generate edge
	srand(size_t(time(00)));
	size_t index_new_node = 0;
	size_t judge_one = 0;
	size_t judge_two = 0;
	
do
{	
	
	graph->AddNode();
	index_new_node = graph->GetNumberOfNodes()-1;
	do
	{
		for(auto node = graph->begin(); (node +1) != graph->end(); node++)
		{
			//random selection
			if(double(rand() % RAND_MAX) / RAND_MAX < probability)
			{	
				graph->AddEdge(node, index_new_node);
				judge_one++;
							
				for(auto second_node= node->begin(); second_node != node->end(); second_node++)
				{
					if((double(rand() % RAND_MAX) / RAND_MAX < probability)&&(*second_node != index_new_node))
					{
						graph->AddEdge(*second_node, index_new_node);
						auto second_contact = graph->find(*second_node);

						for(auto terti_node= second_contact->begin(); terti_node != second_contact->end(); terti_node++)
						{
							if((double(rand() % RAND_MAX) / RAND_MAX < probability)&&(*terti_node != index_new_node))
							{
								graph->AddEdge(*terti_node, index_new_node);
								judge_two++;
							}
						}
					}
				}
			}
		
		}
	
	}while(!(judge_one||judge_two));

}while(graph->GetNumberOfNodes()<numberOfNodes);
	 
 return graph;
}



/************************************************************************
* calculate the society characteristic - Spectral bisection method
* Author: 李辰
* Date: 20100513
************************************************************************/

double scn::RenormalizeBySpectralBisection(UGraph::pGraph graph, size_t length)
{
	
	UGraph::pGraph temp_graph(new UGraph(*graph));
	UNetwork<int>::pNetwork temp_network(new UNetwork<int>(temp_graph));
	UNetwork<int>::pNode temp;
	
	
	double eigenvalue_min_first = 0;
	double eigenvalue_min_second = 0;
	int min_second_index = 0;
	int sizeofgraph = temp_graph->GetNumberOfNodes();

	int group_flag_A = 1;
	int group_flag_B = 2;

	double Q = 0;
	
	auto eigenvalue_list = graph->GetLaplacianMatrix().GetEigenvalueList();

    #undef min
	eigenvalue_min_first = eigenvalue_list.min();
	eigenvalue_min_second = eigenvalue_list[0];
	if(eigenvalue_min_first == eigenvalue_min_second)
	{eigenvalue_min_second = eigenvalue_list[1];}
	min_second_index = 0;

	for(int i = 1; i < sizeofgraph; i++)
	{
		if(eigenvalue_list[i] != eigenvalue_min_first)
		{
			eigenvalue_min_second > eigenvalue_list[i];
			eigenvalue_min_second = eigenvalue_list[i];
			min_second_index = i;
		}

	}

	auto result = temp_graph->GetLaplacianMatrix().GetEigenValueAndVector();

	for(auto node = graph->begin(); node != graph->end(); node++)
	{
		if(result[min_second_index].second[node->GetIndexOfNode()] > 0)
		{
			temp.reset(new int(group_flag_A));
			temp_network->SetNodeData(node,  temp);		
		}else if(result[min_second_index].second[node->GetIndexOfNode()] < 0)
		{
			temp.reset(new int(group_flag_B));
			temp_network->SetNodeData(node,  temp);		
		}
	}

	for(auto node_first = graph->begin(); node_first != graph->end(); node_first++)
	{
		for(auto node_second = graph->begin(); node_second != graph->end(); node_second++)
		{
			if(temp_network->GetNodeData(node_first->GetIndexOfNode()) == temp_network->GetNodeData(node_second->GetIndexOfNode()))
			{
				if(graph->HasEdge(node_first, node_second))
				{
					Q = Q + (1 - (node_first->GetDegree())*(node_second->GetDegree())/(2 * graph->GetNumberOfEdges()));
				}
				else if(!(graph->HasEdge(node_first, node_second)))
				{
					Q = Q - (node_first->GetDegree())*(node_second->GetDegree())/(2 * graph->GetNumberOfEdges());
				}

			}

		}

	}

	Q = Q / (2 * graph->GetNumberOfEdges());


   return Q;
}


	
UGraph::pGraph scn::GenPreferenceMemoryNetwork(size_t numberOfNodes)
{	
	
	//需要设置的参数
	double node_probability = 0.027;
	double event_probability = 0.0458;
	double attraction_attenuation_y = -0.03;
	double activity_attenuation_b = -0.027;
	double degree_coefficient = 0.75;

	int event_response_number_init = 10;

	

	//变量初始化
	
	int active_event_nodeindex = -1;	
	double max_event_attraction = 0;

	int time_step = 0;
	int newNodeAdd_index = 0;
	int newEventAdd_index = 0;


	int active_node_index_first = 0;
	int active_node_index_second = 0;
	int active_node_index_third = 0;
	int term_index = 0;

	double max_activity_first = 0;
	double max_activity_second = 0;
	double max_activity_third = 0;
	double term = 0;

	double compare = 0;

	srand(size_t(time(00)));
	
	struct Node_Attribute
	{
		vector<double> vecInterest;

		double activity;
		int timestep_node_gen;

		double event_attraction;
		int event_flag;
		int event_response_number;
		int timestep_event_gen;

		int group_number;
	};
	
	UGraph::pGraph graph(new UGraph(4));
	UNetwork<Node_Attribute>::pNetwork network(new UNetwork<Node_Attribute>(graph));
		
	for(auto node = graph->begin(); node != graph->end(); node++)
	{
		UNetwork<Node_Attribute>::pNode temp(new Node_Attribute());
		temp->activity = 1;
		temp->event_attraction = 0;
		temp->event_flag = 0;
		temp->event_response_number = 0;
		temp->timestep_node_gen = 0;
		temp->timestep_event_gen = 0;
		

		for(int i=0;i<6;i++)
		{
			temp->vecInterest.push_back(pow(((1- pow(1e-10,3.3))*(double(rand())/double(RAND_MAX))+pow(1e-10,3.3)),0.303030303030));
		}
		network->SetNodeData(node, temp);

		if(node->GetIndexOfNode() == 1)
		{
			network->GetNodeData(node)->event_attraction = 1;
			network->GetNodeData(node)->event_response_number = rand() % event_response_number_init;
			network->GetNodeData(node)->event_flag = 1;
		}

	}
	
	
	do
	{
		time_step++;
		
		//generate new event to a node with event_probability
		if(double(rand() % RAND_MAX) / RAND_MAX < event_probability)
		{
			do
			{active_event_nodeindex = rand() % (graph->GetNumberOfNodes());}
			while(network->GetNodeData(active_event_nodeindex)->event_flag == 1);

			network->GetNodeData(active_event_nodeindex)->event_flag = 1;
			network->GetNodeData(active_event_nodeindex)->event_response_number = rand() % event_response_number_init;
			network->GetNodeData(active_event_nodeindex)->event_attraction = 1;
			network->GetNodeData(active_event_nodeindex)->timestep_event_gen = time_step;
			newEventAdd_index++;
			
		}

		

		//generate new node with node_probability
		if(double(rand() % RAND_MAX) / RAND_MAX < node_probability)
		{
			graph->AddNode();
			active_node_index_first = graph->GetNumberOfNodes()-1;

			UNetwork<Node_Attribute>::pNode temp(new Node_Attribute());
			temp->activity = 1;
			temp->timestep_node_gen = 0;

			temp->event_attraction = 0;
			temp->event_flag = 0;
			temp->event_response_number = 0;			
			temp->timestep_event_gen = 0;


			for(int i=0;i<6;i++)
			{
				temp->vecInterest.push_back(pow(((1- pow(1e-10,3.3))*(double(rand())/double(RAND_MAX))+pow(1e-10,3.3)),0.303030303030));
			}

			network->SetNodeData(active_node_index_first, temp);

			network->GetNodeData(active_node_index_first)->timestep_node_gen = time_step;
			newNodeAdd_index++;
		}


			
		//calculate activity of node and remaining step of event
		for(auto node = graph->begin(); node != graph->end(); node++)
		{
			network->GetNodeData(node->GetIndexOfNode())->activity = exp((time_step - network->GetNodeData(node->GetIndexOfNode())->timestep_node_gen) * activity_attenuation_b);
			if(network->GetNodeData(node)->event_response_number == 0)
			{
				network->GetNodeData(node->GetIndexOfNode())->event_flag = 0;
				network->GetNodeData(node->GetIndexOfNode())->event_attraction = 0;
				network->GetNodeData(node->GetIndexOfNode())->timestep_event_gen = 0;
			}
			if(network->GetNodeData(node->GetIndexOfNode())->event_flag == 1)
			{
				network->GetNodeData(node->GetIndexOfNode())->event_response_number--;
				network->GetNodeData(node->GetIndexOfNode())->event_attraction = exp((time_step - network->GetNodeData(node->GetIndexOfNode())->timestep_event_gen) * attraction_attenuation_y);			
			}

			if(newEventAdd_index == 0 && network->GetNodeData(node->GetIndexOfNode())->event_flag == 1)
			{
				if(max_event_attraction < network->GetNodeData(node->GetIndexOfNode())->event_attraction)
				{
					max_event_attraction = network->GetNodeData(node->GetIndexOfNode())->event_attraction;
					active_event_nodeindex = node->GetIndexOfNode();
				}

			}
			
		}
		


		if(newNodeAdd_index == 0 && active_event_nodeindex != -1)
		{

			max_activity_first = network->GetNodeData(0)->activity * PearsonCoefficient(network->GetNodeData(0)->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
			max_activity_second = network->GetNodeData(2)->activity * PearsonCoefficient(network->GetNodeData(2)->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
			max_activity_third = network->GetNodeData(3)->activity * PearsonCoefficient(network->GetNodeData(3)->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
			
			active_node_index_first = 0;
			active_node_index_second = 2;
			active_node_index_third = 3;

			if(max_activity_second < max_activity_third)
			{
				term = max_activity_second;
				max_activity_second = max_activity_third;
				max_activity_third = term;

				term_index = active_node_index_second;
				active_node_index_second = active_node_index_third;
				active_node_index_third = term_index;
			}
			if(max_activity_first < max_activity_second)
			{
				term = max_activity_first;
				max_activity_first = max_activity_second;
				max_activity_second = term;

				term_index = active_node_index_first;
				active_node_index_first = active_node_index_second;
				active_node_index_second = term_index;
			}
			if(max_activity_second < max_activity_third)
			{
				term = max_activity_second;
				max_activity_second = max_activity_third;
				max_activity_third = term;

				term_index = active_node_index_second;
				active_node_index_second = active_node_index_third;
				active_node_index_third = term_index;
			}



			for(auto node = graph->begin(); node != graph->end(); node++)
			{
				
				if(network->GetNodeData(node->GetIndexOfNode())->event_flag != 1)
				{
					compare = PearsonCoefficient(network->GetNodeData(node->GetIndexOfNode())->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
					if(max_activity_first < (compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges())))
					{
						max_activity_first = compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges());
						active_node_index_first = node->GetIndexOfNode();
					}else if(max_activity_second < (compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges())))
					{
						max_activity_second = compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges());
						active_node_index_second = node->GetIndexOfNode();
					} else if(max_activity_third < (compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges())))
					{
						max_activity_third = compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges());
						active_node_index_third = node->GetIndexOfNode();
					}


				}
			}

		}
		

		if(newNodeAdd_index == 1 && active_event_nodeindex != -1)
		{
			max_activity_second = network->GetNodeData(2)->activity * PearsonCoefficient(network->GetNodeData(2)->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
			max_activity_third = network->GetNodeData(3)->activity * PearsonCoefficient(network->GetNodeData(3)->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
			
			active_node_index_second = 2;
			active_node_index_third = 3;

			if(max_activity_second < max_activity_third)
			{
				term = max_activity_second;
				max_activity_second = max_activity_third;
				max_activity_third = term;

				term_index = active_node_index_second;
				active_node_index_second = active_node_index_third;
				active_node_index_third = term_index;
			}

			for(auto node = graph->begin(); node != graph->end(); node++)
			{
				if(network->GetNodeData(node->GetIndexOfNode())->event_flag != 1)
				{
					compare = PearsonCoefficient(network->GetNodeData(node->GetIndexOfNode())->vecInterest,network->GetNodeData(active_event_nodeindex)->vecInterest);
					if(max_activity_second < (compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges())))
					{
						max_activity_second = compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges());
						active_node_index_second = node->GetIndexOfNode();
					} else if(max_activity_third < (compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges())))
					{
						max_activity_third = compare * network->GetNodeData(node)->activity + node->GetDegree() * degree_coefficient/(graph->GetNumberOfEdges());
						active_node_index_third = node->GetIndexOfNode();
					}

				}

			}




		}

		
		if(newEventAdd_index == 1 || active_event_nodeindex != -1)
		{
			if(!(graph->HasEdge(active_node_index_first, active_event_nodeindex)) && active_node_index_first != active_event_nodeindex)
			{
				graph->AddEdge(active_node_index_first, active_event_nodeindex);
			}
			if(!(graph->HasEdge(active_node_index_second, active_event_nodeindex)) && active_node_index_second != active_event_nodeindex)
			{
				graph->AddEdge(active_node_index_second, active_event_nodeindex);
			}
			if(!(graph->HasEdge(active_node_index_third, active_event_nodeindex)) && active_node_index_third != active_event_nodeindex)
			{
				graph->AddEdge(active_node_index_third, active_event_nodeindex);
			}
			if(!(graph->HasEdge(active_node_index_first, active_node_index_second)) && active_node_index_first != active_node_index_second)
			{
				graph->AddEdge(active_node_index_first, active_node_index_second);
			}
			if(!(graph->HasEdge(active_node_index_first, active_node_index_third)) && active_node_index_first != active_node_index_third)
			{
				graph->AddEdge(active_node_index_first, active_node_index_third);
			}
			if(!(graph->HasEdge(active_node_index_third, active_node_index_second)) && active_node_index_third != active_node_index_second)
			{
				graph->AddEdge(active_node_index_third, active_node_index_second);
			}

		}

		/*
		auto active_node_contact = graph->find(active_node_index);


		for(auto node = active_node_contact->begin(); node != active_node_contact->end(); node++)
		{
			if( !(graph->find(*node)== active_node_contact || graph->HasEdge(*node, active_event_nodeindex)))
			{				
				graph->AddEdge(*node, active_event_nodeindex);
			}
		}

		*/
		
		
		newNodeAdd_index = 0;
		newEventAdd_index = 0;
		max_event_attraction = 0;

		active_event_nodeindex = -1;	

		active_node_index_first = 0;
		active_node_index_second = 0;
		active_node_index_third = 0;
		term_index = 0;
		
		max_activity_first = 0;
		max_activity_second = 0;
		max_activity_third = 0;
		term = 0;

		compare = 0;

	
	}while(graph->GetNumberOfNodes()<numberOfNodes);
	

	return graph;


}


// calculate pearson correlation coefficient
double scn::PearsonCoefficient(vector<double> x, vector<double> y)
{

	double x_ave;
	double y_ave;
	double R;
	double x1,x2,x3,x4,x5,x6;
	double y1,y2,y3,y4,y5,y6;

	x_ave = (x[0]+x[1]+x[2]+x[3]+x[4]+x[5])/6;
	y_ave = (y[0]+y[1]+y[2]+y[3]+y[4]+y[5])/6;
	x1 = x[0]-x_ave;
	x2 = x[1]-x_ave;
	x3 = x[2]-x_ave;
	x4 = x[3]-x_ave;
	x5 = x[4]-x_ave;
	x6 = x[5]-x_ave;
	y1 = y[0]-y_ave;
	y2 = y[1]-y_ave;
	y3 = y[2]-y_ave;
	y4 = y[3]-y_ave;
	y5 = y[4]-y_ave;
	y6 = y[5]-y_ave;
	
	R = (x1*y1+x2*y2+x3*y3+x4*y4+x5*y5+x6*y6)/(sqrt(pow(x1,2)+pow(x2,2)+pow(x3,2)+pow(x4,2)+pow(x5,2)+pow(x6,2))*sqrt(pow(y1,2)+pow(y2,2)+pow(y3,2)+pow(y4,2)+pow(y5,2)+pow(y6,2)));
	
	return R;

}
