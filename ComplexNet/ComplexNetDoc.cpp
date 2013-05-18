
// ComplexNetDoc.cpp : CComplexNetDoc 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "ComplexNet.h"
#endif

#include "ComplexNetDoc.h"

#include <propkey.h>

#include "Network.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CComplexNetDoc

IMPLEMENT_DYNCREATE(CComplexNetDoc, CDocument)

BEGIN_MESSAGE_MAP(CComplexNetDoc, CDocument)
	
END_MESSAGE_MAP()


// CComplexNetDoc 构造/析构

CComplexNetDoc::CComplexNetDoc()
{
	// TODO: 在此添加一次性构造代码
   NetTxtFileOpened=FALSE;
   
   //获得当前工作目录
   GetCurrentDirectory(_MAX_PATH,CurrentWorkPath);

   //获得动态链接库实例
   hInstance = LoadLibrary("NetworkFun.dll");
   
   //获得动态链接库函数接口+++++++++++目前支持无向网络+++++++++++++++++++++++++++++++++++++++开始
   //根据UNetwork结构存储无向dot文件
   WriteToDotFile=(writeToDotFile)GetProcAddress(hInstance,"WriteToDotFile");
   //根据UNetwork结构存储无向bmp文件
   DrawCircleForm=(drawCircleForm)GetProcAddress(hInstance,"DrawCircleForm");
   //根据UNetwork结构存储无向net文件
   WriteToNetFile=(writeToNetFile)GetProcAddress(hInstance,"WriteToNetFile");
   //读取net文件到UNetwork结构
   ReadUNetworkFromNetFile=(readUNetworkFromNetFile)GetProcAddress(hInstance,"ReadUNetworkFromNetFile");
   //规则K最近邻耦合网络生成的函数
   GenKNearestNetwork=(genKNearestNetwork)GetProcAddress(hInstance,"GenKNearestNetwork");
   //随机网络生成的函数
   GenRandomNetwork=(genRandomNetwork)GetProcAddress(hInstance,"GenRandomNetwork");
   //WS小世界网络生成的函数
   GenSmallWorldNetworkByWS=(genSmallWorldNetworkByWS)GetProcAddress(hInstance,"GenSmallWorldNetworkByWS");
   //NW小世界网络生成的函数
   GenSmallWorldNetworkByNW=(genSmallWorldNetworkByNW)GetProcAddress(hInstance,"GenSmallWorldNetworkByNW");
   //BA无标度网络生成的函数
   GenScaleFreeNetwork=(genScaleFreeNetwork)GetProcAddress(hInstance,"GenScaleFreeNetwork");
   //树状小世界确定性网络生成
   GenTreeStructuredSW=(genTreeStructuredSW)GetProcAddress(hInstance,"GenTreeStructuredSW");
   //树状无标度小世界确定性网络生成
   GenTreeStructuredSFSW=(genTreeStructuredSFSW)GetProcAddress(hInstance,"GenTreeStructuredSFSW");


   GenRelationNetwork=(genRelationNetwork)GetProcAddress(hInstance,"GenRelationNetwork");
   GenPreferenceMemoryNetwork=(genPreferenceMemoryNetwork)GetProcAddress(hInstance,"GenPreferenceMemoryNetwork");


   //边迭代小世界网络生成函数
   GenSmallWorldByEdgeIteration=(genSmallWorldByEdgeIteration)GetProcAddress(hInstance,"GenSmallWorldByEdgeIteration");
   //均匀迭代树网络生成函数
   GenUniformRecursiveTree=(genUniformRecursiveTree)GetProcAddress(hInstance,"GenUniformRecursiveTree");
   //确定性均匀迭代树网络生成函数
   GenDURT=(genDURT)GetProcAddress(hInstance,"GenDURT");
   //均匀迭代树小世界网络生成函数
   GenSmallWorldNetworkFromDURT=(genSmallWorldNetworkFromDURT)GetProcAddress(hInstance,"GenSmallWorldNetworkFromDURT");
   //改进边迭代小世界网络生成函数
   GenTriangleExtendedDSWN=(genTriangleExtendedDSWN)GetProcAddress(hInstance,"GenTriangleExtendedDSWN");
   GenSwirlShapedNetwork=(genSwirlShapedNetwork)GetProcAddress(hInstance,"GenSwirlShapedNetwork");
    GenPinWheelShapedSW=(genPinWheelShapedSW)GetProcAddress(hInstance,"GenPinWheelShapedSW");
   //社团网络生成
   GenCommunityNetwork=(genCommunityNetwork)GetProcAddress(hInstance,"GenCommunityNetwork");
   //根据度分布构造网络（暂时无效）
   GenNetworkFromDegreeDistribution=(genNetworkFromDegreeDistribution)GetProcAddress(hInstance,"GenNetworkFromDegreeDistribution");
   IsDegreeListGraphical=(isDegreeListGraphical)GetProcAddress(hInstance,"IsDegreeListGraphical");
   //盒计数法聚类
   RenormalizeByBoxCounting=(renormalizeByBoxCounting)GetProcAddress(hInstance,"RenormalizeByBoxCounting");
   
   RenormalizeBySpectralBisection=(renormalizeBySpectralBisection)GetProcAddress(hInstance,"RenormalizeBySpectralBisection");
   
   //计算平均度
   ComputeAverageDegree=(computeAverageDegree)GetProcAddress(hInstance,"ComputeAverageDegree");
   //计算谱
   ComputeSpectrum=(computeSpectrum)GetProcAddress(hInstance,"ComputeSpectrum");
   //计算同步性能
   GetLambda2AndRatio=(getLambda2AndRatio)GetProcAddress(hInstance,"GetLambda2AndRatio");
   //计算度分布
   GetDegreeDistribution=(getDegreeDistribution)GetProcAddress(hInstance,"GetDegreeDistribution");
   //计算度度相关性
   GetPSDegreeCorrelation=(getPSDegreeCorrelation)GetProcAddress(hInstance,"GetPSDegreeCorrelation");
   //计算Pearson度度相关系数
   GetPearsonCorrCoeff=(getPearsonCorrCoeff)GetProcAddress(hInstance,"GetPearsonCorrCoeff");
    
   //计算谱密度
   ComputeSpectralDensity=(computeSpectralDensity)GetProcAddress(hInstance,"ComputeSpectralDensity");
   //计算度分布熵
   GetEntropyOfDegreeDist=(getEntropyOfDegreeDist)GetProcAddress(hInstance,"GetEntropyOfDegreeDist");
   //计算最短距离
   GetShortestDistance=(getShortestDistance)GetProcAddress(hInstance,"GetShortestDistance");
   //获得最短距离矩阵
   GetGeodesicMatrix=(getGeodesicMatrix)GetProcAddress(hInstance,"GetGeodesicMatrix");
   //计算直径
   GetDiameter=(getDiameter)GetProcAddress(hInstance,"GetDiameter");
   //计算直径和平均最短距离
   GetDiameterAndAverageDistance=(getDiameterAndAverageDistance)GetProcAddress(hInstance,"GetDiameterAndAverageDistance");
   //计算距离分布
   GetShortestDistanceDistribution=(getShortestDistanceDistribution)GetProcAddress(hInstance,"GetShortestDistanceDistribution");
   //计算平均距离
   GetAverageDistanceByDjikstra=(getAverageDistanceByDjikstra)GetProcAddress(hInstance,"GetAverageDistanceByDjikstra");
   //计算平均距离
   GetAverageDistance=(getAverageDistance)GetProcAddress(hInstance,"GetAverageDistance");
   //计算网络全局效率
   GetGlobalEfficiency=(getGlobalEfficiency)GetProcAddress(hInstance,"GetGlobalEfficiency");
   //计算网络脆弱性
   GetVulnerability=(getVulnerability)GetProcAddress(hInstance,"GetVulnerability");
   GetTransitivity=(getTransitivity)GetProcAddress(hInstance,"GetTransitivity");
   //计算集聚系数
   GetClusteringCoeff=(getClusteringCoeff)GetProcAddress(hInstance,"GetClusteringCoeff");
   //计算集聚系数分布
   GetClusteringCoeffDist=(getClusteringCoeffDist)GetProcAddress(hInstance,"GetClusteringCoeffDist");
   //计算聚-度相关性
   GetClusteringDegreeCorre=(getClusteringDegreeCorre)GetProcAddress(hInstance,"GetClusteringDegreeCorre");
   
   //以下函数测试中，等待更新
   GetCyclicCoeff=(getCyclicCoeff)GetProcAddress(hInstance,"GetCyclicCoeff");
   GetRichClubCoeff=(getRichClubCoeff)GetProcAddress(hInstance,"GetRichClubCoeff");
   GetSearchInfo=(getSearchInfo)GetProcAddress(hInstance,"GetSearchInfo");
   GetAverageSearchInfo=(getAverageSearchInfo)GetProcAddress(hInstance,"GetAverageSearchInfo");    
   GetAccessInfo=(getAccessInfo)GetProcAddress(hInstance,"GetAccessInfo");
   GetHideInfo=(getHideInfo)GetProcAddress(hInstance,"GetHideInfo");
   GetBetweennessCentrality=(getBetweennessCentrality)GetProcAddress(hInstance,"GetBetweennessCentrality");
   FindClosureGroup=(findClosureGroup)GetProcAddress(hInstance,"FindClosureGroup");
   RandomWalkByURW=(randomWalkByURW)GetProcAddress(hInstance,"RandomWalkByURW");
   RandomWalkByNRRW=(randomWalkByNRRW)GetProcAddress(hInstance,"RandomWalkByNRRW");
   RandomWalkBySARW=(randomWalkBySARW)GetProcAddress(hInstance,"RandomWalkBySARW");
   MFPTofRandomWalk=(mFPTofRandomWalk)GetProcAddress(hInstance,"MFPTofRandomWalk");
   MRTofRandomWalk=(mRTofRandomWalk)GetProcAddress(hInstance,"MRTofRandomWalk");
   RunDjikstra=(runDjikstra)GetProcAddress(hInstance,"RunDjikstra");
   RunSPFA=(runSPFA)GetProcAddress(hInstance,"RunSPFA");
   GetNumberOfShortestPath=(getNumberOfShortestPath)GetProcAddress(hInstance,"GetNumberOfShortestPath");	
   //获得动态链接库函数接口++++++++++目前支持无向网络+++++++++++++++++++++++++++++++++++++++开始

   //生成NetFiles和Results目录++++++++++++开始
   NetFileTitle="";
   CFileFind a; 
   CString b=CurrentWorkPath;
   b+="//NetFiles//*.*";
   BOOL c;
   if((c=a.FindFile(b))==FALSE)
   {
     b=CurrentWorkPath;
	 b+="//NetFiles";
	 CreateDirectoryA(b,NULL);
   }
   b=CurrentWorkPath;
   b+="//Results//*.*";
   if((c=a.FindFile(b))==FALSE)
   {
     b=CurrentWorkPath;
	 b+="//Results";
	 CreateDirectoryA(b,NULL);
   }
   //生成NetFiles和Results目录++++++++++++结束

   //获得BMP图像信息头
   OpenStandardBMPHeader();
}

CComplexNetDoc::~CComplexNetDoc()
{
}

BOOL CComplexNetDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 在此添加重新初始化代码
	// (SDI 文档将重用该文档)
	if(Success==FALSE)return FALSE;
	return TRUE;
}

//读取BMP信息头，为了存储计算结果的BMP图像
void CComplexNetDoc::OpenStandardBMPHeader()
{
	Success=TRUE;
    long int fh;
    LPBITMAPFILEHEADER bitfile;
    LPBITMAPINFOHEADER bitinfo;
    //open the standard file
	//读取256灰度图像BMP信息头
	fh=_open("standard.dat",_O_RDONLY|_O_BINARY);
	if(fh==-1)
	{
		MessageBox(NULL,"[standard.dat] no found or error!","Initial",MB_ICONSTOP|MB_OK);
		Success=FALSE;
		return;
	}
    //Read file
    if(_read(fh,m_StandardBmpInfo,1078)==-1)
    {
	  _close(fh);   
      MessageBox(NULL,"[standard.dat] read error!","Initial",MB_ICONSTOP|MB_OK);
	  Success=FALSE;
	  return;
    }
    _close(fh);   
    //Read Infomation
	bitfile=(LPBITMAPFILEHEADER)m_StandardBmpInfo;
    bitinfo=(LPBITMAPINFOHEADER)(m_StandardBmpInfo+sizeof(BITMAPFILEHEADER));
    //Judge the information of the standard file
	if(bitfile->bfType!=0x4d42||bitfile->bfOffBits!=1078||bitinfo->biBitCount!=8||bitinfo->biCompression!=0)
	{
		MessageBox(NULL,"[standard.dat] format error!","Initial",MB_ICONSTOP|MB_OK);
		Success=FALSE;
		return;
	}

	//读取彩色图像BMP信息头
	fh=_open("standardc.dat",_O_RDONLY|_O_BINARY);
	if(fh==-1)
	{
		MessageBox(NULL,"[standardc.dat] no found or error!","Initial",MB_ICONSTOP|MB_OK);
		Success=FALSE;
		return;
	}
    //Read file
    if(_read(fh,m_StandardBmpInfoc,54)==-1)
    {
	  _close(fh);   
      MessageBox(NULL,"[standardc.dat] read fail!","Initial",MB_ICONSTOP|MB_OK);
	  Success=FALSE;
	  return;
    }
    _close(fh);   
    //Read Infomation
	bitfile=(LPBITMAPFILEHEADER)m_StandardBmpInfoc;
    bitinfo=(LPBITMAPINFOHEADER)(m_StandardBmpInfoc+sizeof(BITMAPFILEHEADER));
    //Judge the information of the standard file
	if(bitfile->bfType!=0x4d42||bitfile->bfOffBits!=54||bitinfo->biBitCount!=24||bitinfo->biCompression!=0)
	{
		MessageBox(NULL,"[standardc.dat] format error!","Initial",MB_ICONSTOP|MB_OK);
        Success=FALSE;
		return;
	}
}
// CComplexNetDoc 序列化
void CComplexNetDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 在此添加存储代码
		//存储有专门函数，见OnFileSave()和OnFileSaveAs()
	}
	else
	{
		// TODO: 在此添加加载代码
		NetTxtFileOpened=FALSE;
		//Get the file pointer
		CFile *fp;
		//char filenm[_MAX_PATH];//归档文件路径变量
		fp=ar.GetFile();//获得此归档文件的CFile对象指针 
	    ULONGLONG flength;
		flength=fp->GetLength();//获得文件大小
		if(flength<=0)return;
		NetFileName=fp->GetFilePath();//获得文件路径
       
		//读取网络文件到unet结构，目前只支持无向网络
		ReadUNetworkFromNetFile(unet,(char *)NetFileName.GetString());
		if(unet->GetTopology()->GetNumberOfNodes()==0)return;
			
		NetTxtFileOpened=TRUE;
		NetFileTitle=fp->GetFileTitle();
	}
}

#ifdef SHARED_HANDLERS

// 缩略图的支持
void CComplexNetDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 修改此代码以绘制文档数据
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 搜索处理程序的支持
void CComplexNetDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// 从文档数据设置搜索内容。
	// 内容部分应由“;”分隔

	// 例如:  strSearchContent = _T("point;rectangle;circle;ole object;")；
	SetSearchContent(strSearchContent);
}

void CComplexNetDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = NULL;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != NULL)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CComplexNetDoc 诊断

#ifdef _DEBUG
void CComplexNetDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CComplexNetDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

// CComplexNetDoc 命令
void CComplexNetDoc::SetTitle(LPCTSTR lpszTitle)
{
	// TODO: 在此添加专用代码和/或调用基类

	CDocument::SetTitle(lpszTitle);
	CString title;
	if(NetFileTitle=="")
	  lpszTitle="没有网络文件打开";
	else
	{
	  title=lpszTitle;
	  title+=" 网络文件";
	  lpszTitle=(LPCTSTR)title;
	}
	CDocument::SetTitle(lpszTitle);

}
