// ComplexNetDoc.h : CComplexNetDoc 类的接口
//
#pragma once
#include "Network.h"
#include "graph.hpp"
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <utility>
#include<set>
#include<iostream>
#include<fstream>
#include<sstream>
#include <io.h> 
#include <fcntl.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include<queue>

using namespace scn;
#define DEGREEBINS 29
#define CLUSTERBINS 29

//动态连接库"NetworkFun.dll"的函数类型的定义+++++++++++++++++开始
typedef void (*writeToDotFile)(Graph::pGraph graph, std::string filename);
typedef void (*drawCircleForm)(Graph::pGraph graph, std::string filename);
typedef void (*writeToNetFile)(char * path, UNetwork<>::pNetwork &network);
typedef void (*readUNetworkFromNetFile)(UNetwork<>::pNetwork &network, char * path);
typedef UGraph::pGraph (*genKNearestNetwork)(size_t numberOfNodes, size_t k);
typedef UGraph::pGraph (*genSmallWorldNetworkByWS)(size_t numberOfNodes, size_t k,double probability);
typedef UGraph::pGraph (*genSmallWorldNetworkByNW)(size_t numberOfNodes,size_t k,double probability);
typedef UGraph::pGraph (*genTreeStructuredSW)(size_t numberOfNodes);
typedef UGraph::pGraph (*genTreeStructuredSFSW)(size_t numberOfNodes);

typedef UGraph::pGraph (*genRelationNetwork)(size_t numberOfNodes);
typedef UGraph::pGraph (*genPreferenceMemoryNetwork)(size_t numberOfNodes);


typedef double (*communityGirvanNewman)(UGraph::pGraph& graph, vector< vector<size_t>>& CmtyV);


typedef UGraph::pGraph (*genRandomNetwork)(size_t numberOfNodes,double probability);
typedef UGraph::pGraph (*genScaleFreeNetwork)(size_t numberOfNodes, size_t degree);
typedef UGraph::pGraph (*genSmallWorldByEdgeIteration)(size_t times);
typedef UGraph::pGraph (*genUniformRecursiveTree)(size_t numberOfNodes);
typedef UGraph::pGraph (*genDURT)(size_t times);
typedef UGraph::pGraph (*genSmallWorldNetworkFromDURT)(size_t times);
typedef UGraph::pGraph (*genTriangleExtendedDSWN)(size_t times);
typedef UGraph::pGraph (*genPinWheelShapedSW)(size_t times);
typedef UGraph::pGraph (*genSwirlShapedNetwork)(size_t times);
typedef UGraph::pGraph (*genCommunityNetwork)(size_t numberOfNodes, size_t numberOfCommunity,double inner_prob, double outer_prob);
typedef void (*genNetworkFromDegreeDistribution)(pUGraphList &list,std::unordered_map<size_t,size_t> &degree_list);
typedef bool (*isDegreeListGraphical)(std::unordered_map<size_t,size_t> &degree_distribution);
typedef UGraph::pGraph (*renormalizeByBoxCounting)(UGraph::pGraph graph, size_t length);

typedef double (*renormalizeBySpectralBisection)(UGraph::pGraph graph, size_t length);

typedef double (*computeAverageDegree)(UGraph::pGraph graph);
typedef valarray<double> (*computeSpectrum)(UGraph::pGraph graph);
typedef void (*getLambda2AndRatio)(double &lambda2, double &ratio,UGraph::pGraph graph);
typedef void (*getDegreeDistribution)(vector<pair<size_t, size_t>> &pairs,UGraph::pGraph graph);
typedef void (*getPSDegreeCorrelation)(vector<pair<size_t, double>> &correlation,scn::UGraph::pGraph graph);
typedef void (*computeSpectralDensity)(std::unordered_map<double,double> &result,UGraph::pGraph graph,double slide);
typedef double (*getEntropyOfDegreeDist)(UGraph::pGraph graph);
typedef size_t (*getShortestDistance)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget);
typedef Matrix (*getGeodesicMatrix)(UGraph::pGraph graph);
typedef size_t (*getDiameter)(UGraph::pGraph graph);
typedef void (*getDiameterAndAverageDistance)(size_t &Diamter,double &APL, UGraph::pGraph graph);
typedef void (*getShortestDistanceDistribution)(vector<pair<size_t, size_t>> &distribution,UGraph::pGraph graph);
typedef double (*getAverageDistanceByDjikstra)(UGraph::pGraph graph);
typedef double (*getAverageDistance)(UGraph::pGraph graph);
typedef double (*getGlobalEfficiency)(UGraph::pGraph graph);
typedef double (*getVulnerability)(UGraph::pGraph graph,size_t indexOfNode);
typedef double (*getTransitivity)(UGraph::pGraph graph);
typedef double (*getClusteringCoeff)(UGraph::pGraph graph,size_t indexOfNode);
typedef void (*getClusteringCoeffDist)(vector<pair<double, double>> &distribution, scn::UGraph::pGraph graph, double slide);
typedef void (*getClusteringDegreeCorre)(pair<double, vector<pair<size_t, double>>> &correlation,scn::UGraph::pGraph graph);
typedef double (*getCyclicCoeff)(UGraph::pGraph graph,size_t indexOfNode);
typedef double (*getRichClubCoeff)(UGraph::pGraph graph,size_t degree);
typedef double (*getPearsonCorrCoeff)(UGraph::pGraph graph);
typedef double (*getSearchInfo)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget);
typedef double (*getAverageSearchInfo)(UGraph::pGraph graph);    
typedef double (*getAccessInfo)(UGraph::pGraph graph,size_t indexOfNode);
typedef double (*getHideInfo)(UGraph::pGraph graph,size_t indexOfNode);
typedef double (*getBetweennessCentrality)(UGraph::pGraph graph,size_t indexOfNode);
typedef IndexList (*findClosureGroup)(UGraph::pGraph graph,size_t indexOfSeed, size_t length);
typedef size_t (*randomWalkByURW)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget);
typedef size_t (*randomWalkByNRRW)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget);
typedef size_t (*randomWalkBySARW)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget);
typedef size_t (*mFPTofRandomWalk)(UGraph::pGraph graph);
typedef size_t (*mRTofRandomWalk)(UGraph::pGraph graph);
typedef void (*runDjikstra)(UGraph::pGraph graph,size_t indexOfSource,std::unordered_map<size_t,size_t> &distance);
typedef void (*runSPFA)(UGraph::pGraph graph,size_t indexOfSource,std::unordered_map<size_t,size_t> &distance);
typedef std::pair<size_t, size_t> (*getNumberOfShortestPath)(UGraph::pGraph graph,size_t indexOfSource, size_t indexOfTarget,size_t indexOfThrough);

//动态连接库"NetworkFun.dll"的函数类型的定义+++++++++++++++++结束


class CComplexNetDoc : public CDocument
{
protected: // 仅从序列化创建
	CComplexNetDoc();
	DECLARE_DYNCREATE(CComplexNetDoc)

// 特性
public:
	BOOL NetTxtFileOpened;
	char CurrentWorkPath[_MAX_PATH];
	CString	NetFileTitle;
	CString NetFileName;
	NETWORK net;
	UNetwork<>::pNetwork unet;
	//The file headers for 24bits and 256gray-scale image
	unsigned char m_StandardBmpInfo[1078];//256gray
	unsigned char m_StandardBmpInfoc[54];//24bits
    //The initial successful flag
	bool Success;   
	//动态链接库实例
	HINSTANCE hInstance;

	//动态连接库的函数定义++++++++++++++++++开始
	writeToNetFile WriteToNetFile;
	writeToDotFile WriteToDotFile;
	drawCircleForm DrawCircleForm;
    readUNetworkFromNetFile ReadUNetworkFromNetFile;
	genKNearestNetwork GenKNearestNetwork;
    genSmallWorldNetworkByWS GenSmallWorldNetworkByWS;
    genSmallWorldNetworkByNW GenSmallWorldNetworkByNW;
    genTreeStructuredSW GenTreeStructuredSW;
	genTreeStructuredSFSW GenTreeStructuredSFSW;

	genRelationNetwork GenRelationNetwork;
	genPreferenceMemoryNetwork GenPreferenceMemoryNetwork;

	communityGirvanNewman CommunityGirvanNewman;



    genRandomNetwork GenRandomNetwork;
    genScaleFreeNetwork GenScaleFreeNetwork;
    genSmallWorldByEdgeIteration GenSmallWorldByEdgeIteration;
    genUniformRecursiveTree GenUniformRecursiveTree;
    genDURT GenDURT;
    genSmallWorldNetworkFromDURT GenSmallWorldNetworkFromDURT;
	genTriangleExtendedDSWN GenTriangleExtendedDSWN;
	genSwirlShapedNetwork  GenSwirlShapedNetwork;
    genPinWheelShapedSW  GenPinWheelShapedSW;
    genCommunityNetwork GenCommunityNetwork;
    genNetworkFromDegreeDistribution GenNetworkFromDegreeDistribution;
    isDegreeListGraphical IsDegreeListGraphical;
    renormalizeByBoxCounting RenormalizeByBoxCounting;

	renormalizeBySpectralBisection RenormalizeBySpectralBisection;

    computeAverageDegree ComputeAverageDegree;
    computeSpectrum ComputeSpectrum;
    getLambda2AndRatio GetLambda2AndRatio;
    getDegreeDistribution GetDegreeDistribution;
	getPSDegreeCorrelation GetPSDegreeCorrelation;
    computeSpectralDensity ComputeSpectralDensity;
    getEntropyOfDegreeDist GetEntropyOfDegreeDist;
    getShortestDistance GetShortestDistance;
    getGeodesicMatrix GetGeodesicMatrix;
    getDiameter GetDiameter;
    getDiameterAndAverageDistance GetDiameterAndAverageDistance;
	getShortestDistanceDistribution GetShortestDistanceDistribution;
    getAverageDistanceByDjikstra GetAverageDistanceByDjikstra;
    getAverageDistance GetAverageDistance;
    getGlobalEfficiency GetGlobalEfficiency;
    getVulnerability GetVulnerability;
    getTransitivity GetTransitivity;
    getClusteringCoeff GetClusteringCoeff;
	getClusteringCoeffDist GetClusteringCoeffDist;
	getClusteringDegreeCorre GetClusteringDegreeCorre;
    getCyclicCoeff GetCyclicCoeff;
    getRichClubCoeff GetRichClubCoeff;
    getPearsonCorrCoeff GetPearsonCorrCoeff;
    getSearchInfo GetSearchInfo;
    getAverageSearchInfo GetAverageSearchInfo;    
    getAccessInfo GetAccessInfo;
    getHideInfo GetHideInfo;
    getBetweennessCentrality GetBetweennessCentrality;
    findClosureGroup FindClosureGroup;
    randomWalkByURW RandomWalkByURW;
    randomWalkByNRRW RandomWalkByNRRW;
    randomWalkBySARW RandomWalkBySARW;
    mFPTofRandomWalk MFPTofRandomWalk;
    mRTofRandomWalk MRTofRandomWalk;
    runDjikstra RunDjikstra;
    runSPFA RunSPFA;
    getNumberOfShortestPath GetNumberOfShortestPath;
	//动态连接库的函数定义++++++++++++++++++结束
// 操作
public:
	//获得BMP文件头
	void OpenStandardBMPHeader();
// 重写
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// 实现
public:
	virtual ~CComplexNetDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// 用于为搜索处理程序设置搜索内容的 Helper 函数
	void SetSearchContent(const CString& value);
#endif // SHARED_HANDLERS

public:
	virtual void SetTitle(LPCTSTR lpszTitle);
};
