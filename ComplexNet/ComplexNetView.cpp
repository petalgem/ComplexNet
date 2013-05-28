// ComplexNetView.cpp : CComplexNetView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "ComplexNet.h"
#endif

#include "ComplexNetDoc.h"
#include "ComplexNetView.h"
#include "DlgExampleNets.h"
#include "DlgCommuNumSet.h"
#include "RegularDlg.h"
#include "RandomDlg.h"
#include "WSSmallDlg.h"
#include "NWSmallDlg.h"
#include "DeterminDlg.h"
#include "BAModelDlg.h"
#include "DialogDrawCurve.h"
#include "DrawCurve.h"
#include "GroupNetDlg.h"
#include "NetFileSaveAs.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CComplexNetView
IMPLEMENT_DYNCREATE(CComplexNetView, CScrollView)

BEGIN_MESSAGE_MAP(CComplexNetView, CScrollView)
	// 标准打印命令
	ON_COMMAND(ID_FILE_PRINT, &CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CComplexNetView::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_FEATURE_DEGREE, &CComplexNetView::OnFeatureDegree)
	ON_COMMAND(ID_FEATURE_DEGREEDIS, &CComplexNetView::OnFeatureDegreedis)
	ON_COMMAND(ID_FEATURE_DEGREECORR, &CComplexNetView::OnFeatureDegreecorr)
	ON_COMMAND(ID_FEATURE_CLUSTER, &CComplexNetView::OnFeatureCluster)
	ON_COMMAND(ID_FEATURE_AVECLUSTER, &CComplexNetView::OnFeatureAvecluster)
	ON_COMMAND(ID_FEATURE_CLUSTERDIS, &CComplexNetView::OnFeatureClusterdis)
	ON_COMMAND(ID_FEATURE_CLUSTERDEGREE, &CComplexNetView::OnFeatureClusterdegree)
	ON_COMMAND(ID_FEATURE_APL, &CComplexNetView::OnFeatureApl)
	ON_COMMAND(ID_FEATURE_PATHDIS, &CComplexNetView::OnFeaturePathdis)
	ON_COMMAND(ID_FEATURE_APPEIGENVECTOR, &CComplexNetView::OnAppEigenvector)
	ON_COMMAND(ID_FEATURE_CLOSENESS, &CComplexNetView::OnCloseness)
	ON_COMMAND(ID_FEATURE_APPFLOWBETW, &CComplexNetView::OnAppFlowBetw)
	ON_COMMAND(ID_FEATURE_NOMINATION, &CComplexNetView::OnNomination)
	ON_COMMAND(ID_GROUP_KL, &CComplexNetView::OnKernighanLin)
	ON_COMMAND(ID_EXAMPLE_NINENODES, &CComplexNetView::On9nodes)
	ON_COMMAND(ID_EXAMPLE_THIRTEENNODES1, &CComplexNetView::On13nodes)
	ON_COMMAND(ID_EXAMPLE_THIRTEENNODES2, &CComplexNetView::On13nodestwo)
	ON_COMMAND(ID_EXAMPLE_TENNODES, &CComplexNetView::On10nodes)
	ON_COMMAND(ID_EXAMPLE_SEXNETWORK, &CComplexNetView::OnSexnetwork)
	ON_COMMAND(ID_EXAMPLE_KARATENETWORK, &CComplexNetView::OnKarate)
	ON_COMMAND(ID_NET_REGULAR, &CComplexNetView::OnNewRegular)
	ON_COMMAND(ID_NET_RANDOM, &CComplexNetView::OnNetRandom)
	ON_COMMAND(ID_NET_WSSMALL, &CComplexNetView::OnNetWssmall)
	ON_COMMAND(ID_NET_NWSMALL, &CComplexNetView::OnNetNWSmall)
	ON_COMMAND(ID_NET_BA, &CComplexNetView::OnNetBA)
	ON_COMMAND(ID_DETERMIN_EI, &CComplexNetView::OnDeterminswei)
	ON_COMMAND(ID_DETERMIN_URT, &CComplexNetView::OnDeterminUrt)
	ON_COMMAND(ID_DETERMIN_SWURT, &CComplexNetView::OnDeterminSwurt)
	ON_COMMAND(ID_DETERMIN_PINWHEEL, &CComplexNetView::OnDeterminPinwheel)
	ON_COMMAND(ID_DETERMIN_SWIRL, &CComplexNetView::OnDeterminSwirl)
	ON_COMMAND(ID_DETERMIN_TSSFSW, &CComplexNetView::OnDeterminTssfsw)
	ON_COMMAND(ID_DETERMIN_TS, &CComplexNetView::OnDeterminTs)
	ON_COMMAND(ID_DETERMIN_MEI, &CComplexNetView::OnDeterminMei)
	ON_COMMAND(ID_GROUP_BOXCOUNTING, &CComplexNetView::OnBoxCounting)
	ON_COMMAND(ID_GROUPNET_SIMPLE, &CComplexNetView::OnGroupnetSimple)
	ON_COMMAND(ID_SYNC_LAMDA2, &CComplexNetView::OnSyncLamda2)
	ON_COMMAND(ID_FILE_SAVE, &CComplexNetView::OnFileSave)
	ON_COMMAND(ID_FILE_SAVE_AS, &CComplexNetView::OnFileSaveAs)
	ON_COMMAND(ID_FEATURE_AVEDEGREE, &CComplexNetView::OnFeatureAvedegree)
	ON_COMMAND(ID_NET_URT, &CComplexNetView::OnNetUrt)
	ON_COMMAND(ID_NET_RELATION, &CComplexNetView::OnNetRelation)
	ON_COMMAND(ID_NET_PMNETWORK, &CComplexNetView::OnNetPmnetwork)
	ON_COMMAND(ID_GROUP_SPECTRUM, &CComplexNetView::OnGroupSpectrum)
	ON_COMMAND(ID_GROUP_GN, &CComplexNetView::OnGroupGn)
END_MESSAGE_MAP()

// CComplexNetView 构造/析构
CComplexNetView::CComplexNetView()
{
	// TODO: 在此处添加构造代码

}

CComplexNetView::~CComplexNetView()
{
}

BOOL CComplexNetView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式

	return CScrollView::PreCreateWindow(cs);
}

// CComplexNetView 绘制
void CComplexNetView::OnDraw(CDC* pDC)
{
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	//+++++++++++++++我们的代码++++++++++++开始
	if (!pDoc)
		return;
	CBrush cb,cb1,*cbold;//The Current brush and the old brush
	CPen cp,*cpold,cp1;
	long int a,b;
	a =GetSystemMetrics(SM_CXFULLSCREEN);
    b = GetSystemMetrics(SM_CYFULLSCREEN);
	 
	  
	////If the image size is not right, set the screen dark
	if(pDoc->NetTxtFileOpened==FALSE)
	{
	  cb.CreateSolidBrush(RGB(128,128,128));
	  cbold=pDC->SelectObject(&cb);
	  pDC->Rectangle(0,0,a,b);
	  pDC->SelectObject(cbold);
	  return;
	}
	
	////Else clear the user screen in white
	cb.CreateSolidBrush(RGB(255,255,255));
	cbold=pDC->SelectObject(&cb);
	pDC->Rectangle(0,0,a,b);
	pDC->SelectObject(cbold);//Restore the old Brush
	//Delete the brush object cb
	::DeleteObject(&cb);

	pDC->SetTextColor(RGB(255,0,0));
    CString	tempstr=pDoc->GetTitle();
	char temp[256];
	if(pDoc->net.Directed==true)
	  tempstr+=" (Directed&";
	else
	  tempstr+=" (Undirected&";
	if(pDoc->net.Weighted==true)
	  tempstr+="Weighted)";
	else
	  tempstr+="Unweighted)";
	sprintf_s(temp,"%s; Nodes: %d; Links: %d",tempstr,pDoc->unet->GetTopology()->GetNumberOfNodes(),pDoc->unet->GetTopology()->GetNumberOfEdges());
	pDC->TextOut(10,20,temp);
	pDC->SetTextColor(RGB(0,0,255));
	//display the nodes
    cp.CreatePen(PS_SOLID,1,RGB(0,0,255));
	cpold=pDC->SelectObject(&cp);
	cb1.CreateSolidBrush(RGB(0,0,255));
	cbold=pDC->SelectObject(&cb1);
	if(pDoc->unet->GetTopology()->GetNumberOfNodes()<4000)
	{
	  auto graph=pDoc->unet->GetTopology();
	  for(auto node=graph->begin();node!=graph->end();node++)
	  {
        auto posi=pDoc->unet->GetNodePosition(*node);
		if(posi[0]<=1&&posi[0]>=0&&posi[1]<=1&&posi[1]>=0)
		{
		   pDC->Ellipse((long int)(posi[0]*800.0)+20,(long int)(posi[1]*600.0)+40,(long int)(posi[0]*800.0)+30,(long int)(posi[1]*600.0)+50);
		   sprintf_s(temp,"%d",*node);
		   pDC->TextOut((long int)(posi[0]*800.0)+25,(long int)(posi[1]*600.0)+50,temp);
		}
	  } 
	}
    
	pDC->SelectObject(cbold);//Restore the old Brush
	//Delete the brush object cb
	::DeleteObject(&cb1);
	pDC->SelectObject(cpold);//Restore the old Brush
	//Delete the brush object cb
	::DeleteObject(&cp);

	cp1.CreatePen(PS_SOLID,1,RGB(255,0,0));
	cpold=pDC->SelectObject(&cp1);
	long int a1,a2,b1,b2;
	if(pDoc->unet->GetTopology()->GetNumberOfNodes()<4000&&pDoc->unet->GetTopology()->GetNumberOfEdges()<10000)
	{
       auto graph=pDoc->unet->GetTopology();
	   for(auto node=graph->begin();node!=graph->end();node++)
	   {
		  auto posi=pDoc->unet->GetNodePosition(*node);
		  for(auto othernd=node->begin();othernd!=node->end();othernd++)
		  {
	          auto posi2=pDoc->unet->GetNodePosition(*othernd);
			  a1=(long int)(posi[0]*800.0)+25;
	          a2=(long int)(posi2[0]*800.0)+25;
	          b1=(long int)(posi[1]*600.0)+45;
	          b2=(long int)(posi2[1]*600.0)+45;
	          pDC->MoveTo(a1,b1);
		      pDC->LineTo(a2,b2);
		  }
	  }
	}	
	pDC->SelectObject(cpold);//Restore the old Brush
	//Delete the brush object cb
	::DeleteObject(&cp);
   //+++++++++++++++我们的代码++++++++++++结束


	// TODO: 在此处为本机数据添加绘制代码
}

void CComplexNetView::OnInitialUpdate()
{
	//+++++++++++++++我们的代码++++++++++++开始
	//初始化面板大小为整个屏幕
	CSize sizeTotal;
	sizeTotal.cx =GetSystemMetrics(SM_CXFULLSCREEN);
	sizeTotal.cy = GetSystemMetrics(SM_CYFULLSCREEN);
	SetScrollSizes(MM_TEXT,sizeTotal);
	//+++++++++++++++我们的代码++++++++++++结束
	CScrollView::OnInitialUpdate();
}


// CComplexNetView 打印


void CComplexNetView::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CComplexNetView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 默认准备
	return DoPreparePrinting(pInfo);
}

void CComplexNetView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 添加额外的打印前进行的初始化过程
}

void CComplexNetView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 添加打印后进行的清理过程
}

void CComplexNetView::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CComplexNetView::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// CComplexNetView 诊断

#ifdef _DEBUG
void CComplexNetView::AssertValid() const
{
	CScrollView::AssertValid();
}

void CComplexNetView::Dump(CDumpContext& dc) const
{
	CScrollView::Dump(dc);
}

CComplexNetDoc* CComplexNetView::GetDocument() const // 非调试版本是内联的
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CComplexNetDoc)));
	return (CComplexNetDoc*)m_pDocument;
}
#endif //_DEBUG


// CComplexNetView 消息处理程序


//NetworkFun.dll度计算不需要专门函数，只需unet本身的类函数
//平均度函数ComputeAverageDegree来自NetworkFun.dll
void CComplexNetView::OnFeatureDegree()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	
    char temp[256];//For string generation
    CString bookstr,txtname;//For string written
	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\degrees.txt";
    //Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
	
    //Save file
    sprintf_s(temp,"%s     %s\r\n","Node","Degree");
    
    bookstr=temp;
    fh.Write(bookstr,bookstr.GetLength());
    bookstr="";
   //Only for Undirected network now
   auto graph=pDoc->unet->GetTopology();
   vector<pair<size_t,size_t>> templist;
   for(auto node=graph->begin();node!=graph->end();node++)
   {
	templist.push_back(make_pair(*node,node->GetDegree()));
   }
   sort(templist.begin(),templist.end());
   for(auto node=templist.begin();node!=templist.end();node++)
   {
     sprintf_s(temp,"%u ",node->first);
	 bookstr+=temp;
     sprintf_s(temp,"%u\r\n",node->second);
	 bookstr+=temp;
  }
  fh.Write(bookstr,bookstr.GetLength());//write string
  bookstr="";
  sprintf_s(temp,"%s     %.2f\r\n","Average Degree=",pDoc->ComputeAverageDegree(graph));
  bookstr=temp;
  fh.Write(bookstr,bookstr.GetLength());
  //close the file 
  fh.Close();
  MessageBox(_T("度计算结果已存到文件degrees.txt"),_T("度计算结果"),MB_OK|MB_ICONINFORMATION);
}

//计算显示平均度，函数ComputeAverageDegree来自NetworkFun.dll
void CComplexNetView::OnFeatureAvedegree()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	char temp[256];
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	sprintf_s(temp,"%s     %.6f\r\n","Average Degree=",pDoc->ComputeAverageDegree(pDoc->unet->GetTopology()));
    MessageBox(temp);
}

//计算和显示度分布，函数GetDegreeDistribution来自NetworkFun.dll
void CComplexNetView::OnFeatureDegreedis()
{
	// TODO: 在此添加命令处理程序代码CNetAllDoc* pDoc = GetDocument();
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	vector<pair<size_t, size_t>> pairs;
	size_t total=pDoc->unet->GetTopology()->GetNumberOfNodes();
	pDoc->GetDegreeDistribution(pairs,pDoc->unet->GetTopology());
	double *ppp;
	double *kkk;
	ppp=new double [total];
	if(ppp==NULL)
	{
		MessageBox(_T("Memory Error"));
		return;
	}
	kkk=new double [total];
	if(kkk==NULL)
	{
		MessageBox(_T("Memory Error"));
		delete []ppp;
		return;
	}
		
	//Gnuplot plot;
    //plot.set_title("degree distribution");
    //plot.plot_equation("sin(x)");
	//plot.savetopng("sin.png");
	//plot.plot_histograms(pairs);
    //plot.showonscreen();
	
    char temp[256];//For string generation
    CString bookstr,txtname;//For string written
	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\p(k).txt";
    //Open the file in WriteOnly Mode
    CFile fh(txtname.GetBuffer(txtname.GetLength()),CFile::modeCreate|CFile::modeWrite);
    //Save file
    sprintf_s(temp,"%s %s\r\n","K","P(K)");
    bookstr=temp;
    fh.Write(bookstr,bookstr.GetLength());
    bookstr="";
    size_t length=0;
    for(auto x=pairs.begin();x!=pairs.end();x++)
    {
	  kkk[length]=x->first;
	  sprintf_s(temp,"%f     ",kkk[length]);
	  bookstr+=temp;
	  ppp[length]=(double)(x->second)/total;
	  sprintf_s(temp,"%.4f\r\n",ppp[length]);
	  bookstr+=temp;	 
	  length++;
   }
   fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("度分布计算结果已存到文件p(k).txt"),_T("度分布计算结果"),MB_OK|MB_ICONINFORMATION);
   
   CDialogDrawCurve a;
   CDrawCurve c;
   a.m_lNumberofSamples=length;
   a.m_bSampleDimensions=2;
   a.m_dDataX=kkk;
   c.m_DataX=a.m_dDataX;
   a.m_dDataY=ppp;
   c.m_DataY=a.m_dDataY;
   c.m_Dimension=2;
   c.m_Length=length;
   if(c.SetDefaultCDF())
   {
	  c.GetCDFData(&a.cdf);
	  strcpy_s(a.cdf.m_sTitles[0],"Degree Distribution Histogram");
	  strcpy_s(a.cdf.m_sTitles[1],"Fig.1 Degree Distribution Histogram");
	  strcpy_s(a.cdf.m_sTitles[2],"Degree");
	  strcpy_s(a.cdf.m_sTitles[3],"Percent");
	  a.DoModal();
   }
   delete []ppp;
   delete []kkk;
   Invalidate(FALSE);
}

//计算度相关性，函数GetPSDegreeCorrelation来自NetworkFun.dll
//计算相关系数，函数GetPearsonCorrCoeff来自NetworkFun.dll
void CComplexNetView::OnFeatureDegreecorr()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	char temp[256];//For string generation
    sprintf_s(temp,"%s     %.6f\r\n","Pearson Correlation Coefficient=",pDoc->GetPearsonCorrCoeff(pDoc->unet->GetTopology()));
	MessageBox(temp);
	vector<pair<size_t,double>> corre;
	pDoc->GetPSDegreeCorrelation(corre,pDoc->unet->GetTopology());
	double *knn;
	double *kkk;
	knn=new double [corre.size()];
	if(knn==NULL)
	{
		MessageBox(_T("Memory Error"));
		return;
	}
	kkk=new double [corre.size()];
	if(kkk==NULL)
	{
		MessageBox(_T("Memory Error"));
		delete []knn;
		return;
	}
	
    
    CString bookstr,txtname;//For string written
    txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\knn(k).txt";
    //Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","K","Knn");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";
   size_t ind=0;
   for(auto k=corre.begin();k!=corre.end();k++)
   {
	 kkk[ind]=k->first;
	 sprintf_s(temp,"%.2f     ",kkk[ind]);
	 bookstr+=temp;
	 knn[ind]=k->second;
     sprintf_s(temp,"%.2f\r\n",knn[ind]);
	 bookstr+=temp;
	 ind++;
   }
   fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("度-度相关计算结果已存到文件knn(k).txt"),_T("度相关计算结果"),MB_OK|MB_ICONINFORMATION);

   CDialogDrawCurve a;
   CDrawCurve c;
   a.m_lNumberofSamples=corre.size();
   a.m_bSampleDimensions=2;
   a.m_dDataX=kkk;
   c.m_DataX=a.m_dDataX;
   a.m_dDataY=knn;
   c.m_DataY=a.m_dDataY;
   c.m_Dimension=1;
   c.m_Length=corre.size();
   if(c.SetDefaultCDF())
   {
	  c.GetCDFData(&a.cdf);
	  strcpy_s(a.cdf.m_sTitles[0],"Degree-degree Correlation");
	  strcpy_s(a.cdf.m_sTitles[1],"Fig.1 Degree-degree Correlation");
	  strcpy_s(a.cdf.m_sTitles[2],"k");
	  strcpy_s(a.cdf.m_sTitles[3],"knn");
	  a.DoModal();
   }
   Invalidate(FALSE);
   delete []knn;
   delete []kkk;
}

//计算各节点集聚系数,函数GetClusteringCoeff来自NetworkFun.dll
void CComplexNetView::OnFeatureCluster()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	char temp[256];//For string generation
    CString bookstr,txtname;//For string written
    txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\clustercoeff.txt";
    //Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);

    sprintf_s(temp,"%s     %s\r\n","Node","ClusterCoeff");
    bookstr=temp;
    fh.Write(bookstr,bookstr.GetLength());
    bookstr="";
	UGraph::pGraph gra=pDoc->unet->GetTopology();
	
	vector<pair<size_t,double>> templist;
	for(auto ii=gra->begin();ii!=gra->end();ii++)
    {
		templist.push_back(make_pair(*ii,pDoc->GetClusteringCoeff(pDoc->unet->GetTopology(),*ii)));
    }
	sort(templist.begin(),templist.end());
    double total=0.0,single;
	size_t number=0;
	for(auto ii=templist.begin();ii!=templist.end();ii++)
    {
	 sprintf_s(temp,"%u     ",ii->first);
	 bookstr+=temp;
	 sprintf_s(temp,"%.6f\r\n",ii->second);
	 bookstr+=temp;
	 total=total+ii->second;
	 number++;
    }
    fh.Write(bookstr,bookstr.GetLength());//write string
    bookstr="";
    sprintf_s(temp,"%s     %.6f\r\n","Average",pDoc->GetClusteringCoeff(pDoc->unet->GetTopology(),NaF));
	bookstr=temp;
    fh.Write(bookstr,bookstr.GetLength());
    //close the file 
    fh.Close();
    MessageBox(_T("集聚系数计算结果已存到文件clustercoeff.txt"),_T("度计算结果"),MB_OK|MB_ICONINFORMATION);
}

//计算平均集聚系数,函数GetClusteringCoeff来自NetworkFun.dll，第二个参数送入NaF
void CComplexNetView::OnFeatureAvecluster()
{
	// TODO: 在此添加命令处理程序代码
	char temp[256];//For string generation
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
    sprintf_s(temp,"%s     %.6f\r\n","Network Clustering Coefficient=",pDoc->GetClusteringCoeff(pDoc->unet->GetTopology(),NaF));
    MessageBox(temp);
}

//计算集聚系数分布,函数GetClusteringCoeffDis来自NetworkFun.dll
void CComplexNetView::OnFeatureClusterdis()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	vector<pair<double, double>> distribution;
	pDoc->GetClusteringCoeffDist(distribution,pDoc->unet->GetTopology(),0);
	double *ppp;
	double *ccc;
	ppp=new double [distribution.size()];
	if(ppp==NULL)
	{
		MessageBox(_T("Memory Error"));
		return;
	}
	ccc=new double [distribution.size()];
	if(ccc==NULL)
	{
		MessageBox(_T("Memory Error"));
		delete []ppp;
		return;
	}
    
    char temp[256];//For string generation
    CString bookstr,txtname;//For string written
    
    txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\p(c).txt";
    //Open the file in WriteOnly Mode
     CFile fh(txtname.GetBuffer(txtname.GetLength()),CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","C","P(C)");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";
   size_t ind=0;
   for(auto k=distribution.begin();k!=distribution.end();k++)
   {
	 ccc[ind]=k->first;
	 sprintf_s(temp,"%.2f     ",ccc[ind]);
	 bookstr+=temp;
	 ppp[ind]=k->second;
     sprintf_s(temp,"%.2f\r\n",ppp[ind]);
	 bookstr+=temp;
	 ind++;
   }
   fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("集聚系数分布计算结果已存到文件p(c).txt"),_T("集聚系数分布计算结果"),MB_OK|MB_ICONINFORMATION);
   
   CDialogDrawCurve a;
	CDrawCurve c;
	a.m_lNumberofSamples=distribution.size();
	a.m_bSampleDimensions=2;
	a.m_dDataX=ccc;
	c.m_DataX=a.m_dDataX;
	a.m_dDataY=ppp;
	c.m_DataY=a.m_dDataY;
	c.m_Dimension=1;
	c.m_Length=distribution.size();
	if(c.SetDefaultCDF())
	{
	  c.GetCDFData(&a.cdf);
	  strcpy_s(a.cdf.m_sTitles[0],"Clustering Coeff Distribution");
	  strcpy_s(a.cdf.m_sTitles[1],"Fig.1 Clustering Coeff Distribution");
	  strcpy_s(a.cdf.m_sTitles[2],"Clustering Coefficients");
	  strcpy_s(a.cdf.m_sTitles[3],"Percent");
	  a.DoModal();
	}
	delete []ppp;
    delete []ccc;
    Invalidate(FALSE);
}

//计算聚度相关性,函数GetClusteringCoeffDis来自NetworkFun.dll
void CComplexNetView::OnFeatureClusterdegree()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	pair<double, vector<pair<size_t, double>>> correlation;
	pDoc->GetClusteringDegreeCorre(correlation,pDoc->unet->GetTopology());
	double *cnn;
	double *kkk;
	cnn=new double [correlation.second.size()];
	if(cnn==NULL)
	{
		MessageBox(_T("Memory Error"));
		return;
	}
	kkk=new double [correlation.second.size()];
	if(kkk==NULL)
	{
		MessageBox(_T("Memory Error"));
		delete []cnn;
		return;
	}
    char temp[256];//For string generation
    CString bookstr,txtname;//For string written
    
    txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\cnn(k).txt";
    //Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","K","Cnn");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";
   size_t ind=0;
   for(auto k=correlation.second.begin();k!=correlation.second.end();k++)
   {
	 kkk[ind]=k->first;
	 sprintf_s(temp,"%.2f     ",kkk[ind]);
	 bookstr+=temp;
	 cnn[ind]=k->second;
     sprintf_s(temp,"%.2f\r\n",cnn[ind]);
	 bookstr+=temp;
	 ind++;
   }
   fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("聚-度相关计算结果已存到文件cnn(k).txt"),_T("聚-度相关计算结果"),MB_OK|MB_ICONINFORMATION);
   CDialogDrawCurve a;
   CDrawCurve c;
   a.m_lNumberofSamples=correlation.second.size();
   a.m_bSampleDimensions=2;
   a.m_dDataX=kkk;
   c.m_DataX=a.m_dDataX;
   a.m_dDataY=cnn;
   c.m_DataY=a.m_dDataY;
   c.m_Dimension=1;
   c.m_Length=correlation.second.size();
   if(c.SetDefaultCDF())
   {
	  c.GetCDFData(&a.cdf);
	  strcpy_s(a.cdf.m_sTitles[0],"Cluster-degree Correlation");
	  strcpy_s(a.cdf.m_sTitles[1],"Fig.1 Cluster-degree Correlation");
	  strcpy_s(a.cdf.m_sTitles[2],"k");
	  strcpy_s(a.cdf.m_sTitles[3],"cnn");
	  a.DoModal();
   }
   delete []cnn;
   delete []kkk;
   
   Invalidate(FALSE);
}

//计算平均距离和直径，函数GetDiameterAndAverageDistance来自NetworkFun.dll
void CComplexNetView::OnFeatureApl()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	char temp[256];
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	double apl;
	size_t diam;
	pDoc->GetDiameterAndAverageDistance(diam,apl,pDoc->unet->GetTopology());
    sprintf_s(temp,"The average distance: %f; The Diameter: %d",apl,diam);
    MessageBox(temp);

}

//计算距离分布,函数GetShortestDistanceDistribution来自NetworkFun.dll
void CComplexNetView::OnFeaturePathdis()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	char temp[256];
	double *L;
	double *PL;
	size_t diameter=pDoc->GetDiameter(pDoc->unet->GetTopology());
	L=new double [diameter];
	PL=new double [diameter];
	CString bookstr,txtname;//For string written
    vector<pair<size_t,size_t>> distribution;
	size_t total=pDoc->unet->GetTopology()->GetNumberOfNodes();
	total=total*(total-1);
	pDoc->GetShortestDistanceDistribution(distribution,pDoc->unet->GetTopology());
	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\p(L).txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
	//Save file
    sprintf_s(temp,"%s %s\r\n","L","P(L)");
    bookstr=temp;
    fh.Write(bookstr,bookstr.GetLength());
    bookstr="";
    diameter=0;
	for(auto ii=distribution.begin();ii!=distribution.end();ii++)
	{	
		L[diameter]=ii->first;
		PL[diameter]=ii->second/(double)total;
		sprintf_s(temp,"%f %.6f  \r\n",L[diameter],PL[diameter]);
		bookstr+=temp;
		diameter++;
	}
	
   fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("距离分布计算结果已存到文件p(L).txt"),_T("距离分布计算结果"),MB_OK|MB_ICONINFORMATION);
   
   CDialogDrawCurve a;
   CDrawCurve c;
   a.m_lNumberofSamples=diameter;
   a.m_bSampleDimensions=2;
   a.m_dDataX=L;
   c.m_DataX=a.m_dDataX;
   a.m_dDataY=PL;
   c.m_DataY=a.m_dDataY;
   c.m_Dimension=2;
   c.m_Length=diameter;
   if(c.SetDefaultCDF())
   {
	  c.GetCDFData(&a.cdf);
	  strcpy_s(a.cdf.m_sTitles[0],"Distance Distribution Histogram");
	  strcpy_s(a.cdf.m_sTitles[1],"Fig.1 Distance Distribution Histogram");
	  strcpy_s(a.cdf.m_sTitles[2],"Distance");
	  strcpy_s(a.cdf.m_sTitles[3],"Percent");
	  a.DoModal();
   }
   delete []L;
   delete []PL;
}

//目前该功能不好使，等待更新
void CComplexNetView::OnAppEigenvector()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	MessageBox("目前该功能不好使，等待更新");
	/*double *feature;
	double *temptemp;
	feature=new double [pDoc->net.NodeNumber];
	temptemp=new double [pDoc->net.NodeNumber];
		
	pDoc->CalculateNodesCentralityAppEigenvector(&pDoc->net,feature,temptemp);
	
    char temp[256];//For string generation
	WORD k;
    CString bookstr,txtname;//For string written

	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\AppEigenvector.txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","Node","AppEigenvector");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";	
   for(k=0;k<pDoc->net.NodeNumber;k++)
   {
     sprintf_s(temp,"%u ",pDoc->net.Net[k].NodeNo);
	 bookstr+=temp;     
	 sprintf_s(temp,"%.6f \r\n",feature[k]);
	 bookstr+=temp;	
   }    

	fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
	fh.Close();
    MessageBox(_T("近似特征向量指标计算结果已存到文件AppEigenvector.txt"),_T("近似特征向量指标计算结果"),MB_OK|MB_ICONINFORMATION);
	delete [] feature;
	delete [] temptemp;*/
}

//目前该功能不好使，等待更新
void CComplexNetView::OnCloseness()
{
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	MessageBox("目前该功能不好使，等待更新");
/*	double *feature;
	feature=new double [pDoc->net.NodeNumber];
	pDoc->CalculateNodesCentralityCloseness(&pDoc->net,feature);
	
    char temp[256];//For string generation
	WORD k;
    CString bookstr,txtname;//For string written

	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\Closeness.txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","Node","Closeness");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";	
   for(k=0;k<pDoc->net.NodeNumber;k++)
   {
     sprintf_s(temp,"%u ",pDoc->net.Net[k].NodeNo);
	 bookstr+=temp;     
	 sprintf_s(temp,"%.6f \r\n",feature[k]);
	 bookstr+=temp;	
   }
    
	fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
   fh.Close();
   MessageBox(_T("紧密度指标计算结果已存到文件Closeness.txt"),_T("紧密度指标计算结果"),MB_OK|MB_ICONINFORMATION);	
	delete [] feature;*/
	
}
//目前该功能不好使，等待更新
void CComplexNetView::OnAppFlowBetw()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	MessageBox("目前该功能不好使，等待更新");
/*	double *feature;
	double *temptemp;
	feature=new double [pDoc->net.NodeNumber];
	temptemp=new double [pDoc->net.NodeNumber];
	
	pDoc->CalculateNodesCentralityAppFlowBetweenness(&pDoc->net,feature,temptemp);
	
    char temp[256];//For string generation
	WORD k;
    CString bookstr,txtname;//For string written

	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\AppFlowBetweenness.txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","Node","AppFlowBetweenness");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";	
   for(k=0;k<pDoc->net.NodeNumber;k++)
   {
     sprintf_s(temp,"%u ",pDoc->net.Net[k].NodeNo);
	 bookstr+=temp;     
	 sprintf_s(temp,"%.6f \r\n",feature[k]);
	 bookstr+=temp;	
   }      
	fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
	fh.Close();
   MessageBox(_T("近似流介数指标计算结果已存到文件AppFlowBetweenness.txt"),_T("近似流介数指标计算结果"),MB_OK|MB_ICONINFORMATION);
	delete [] feature;
	delete [] temptemp;*/
}
//目前该功能不好使，等待更新
void CComplexNetView::OnNomination()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	MessageBox("目前该功能不好使，等待更新");
	/*double *feature;
	double *temptemp;
	double *temp1;
	feature=new double [pDoc->net.NodeNumber];
	temptemp=new double [pDoc->net.NodeNumber];
	temp1=new double [pDoc->net.NodeNumber];
	
	pDoc->CalculateNodesCentralityNomination(&pDoc->net,feature,temptemp,temp1);
	
    char temp[256];//For string generation
	WORD k;
    CString bookstr,txtname;//For string written

	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\Nomination.txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","Node","Nomination");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   bookstr="";	
   for(k=0;k<pDoc->net.NodeNumber;k++)
   {
     sprintf_s(temp,"%u ",pDoc->net.Net[k].NodeNo);
	 bookstr+=temp;     
	 sprintf_s(temp,"%.6f \r\n",feature[k]);
	 bookstr+=temp;	
   }    

	fh.Write(bookstr,bookstr.GetLength());//write string
   //close the file 
	fh.Close();
   MessageBox(_T("累计提名指标计算结果已存到文件Nomination.txt"),_T("累计提名指标计算结果"),MB_OK|MB_ICONINFORMATION);
   	delete [] feature;
	delete [] temptemp;
	delete [] temp1;*/
}

//目前该功能不好使，等待更新
void CComplexNetView::OnKernighanLin()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	MessageBox("目前该功能不好使，等待更新");
	/*CDlgCommuNumSet dr;
	dr.DoModal();
	int *comlist;
	comlist=new int [pDoc->net.NodeNumber];
	
	if(pDoc->CalculateNetworkCommunityKernighanLin(pDoc->net,dr.m_Commu_One_Num,dr.m_Commu_Two_Num,comlist)!=0)
	{
      MessageBox(_T("Wrong input paramete!!"));
	  return;	
	}
    char temp[256];//For string generation
	WORD k;
    CString bookstr,txtname,bookstrA,bookstrB;//For string written
	txtname=pDoc->CurrentWorkPath;
	txtname+="\\Results\\KernihanLin.txt";
	//Open the file in WriteOnly Mode
    CFile fh(txtname,CFile::modeCreate|CFile::modeWrite);
   //Save file
   sprintf_s(temp,"%s %s\r\n","Node","KernighanLin");
   bookstr=temp;
   fh.Write(bookstr,bookstr.GetLength());
   sprintf_s(temp,"%s %s \r\n","Community","No. one");
   bookstrA=temp;
   sprintf_s(temp,"%s %s \r\n","Community","No. two");
   bookstrB=temp;	
   for(k=0;k<pDoc->net.NodeNumber;k++)
   {
	   if(comlist[k])
	   {
	    sprintf_s(temp,"%u \r\n",pDoc->net.Net[k].NodeNo);
		bookstrA+=temp; 
	   }
	   else
	   {
		   sprintf_s(temp,"%u \r\n",pDoc->net.Net[k].NodeNo);
		   bookstrB+=temp;	
	   
	   }
   }    
	fh.Write(bookstrA,bookstrA.GetLength());//write string
	fh.Write(bookstrB,bookstrB.GetLength());//write string
   //close the file 
	fh.Close();
    MessageBox(_T("KernighanLin分析结果已存到文件KernighanLin.txt"),_T("KernighanLin分析结果"),MB_OK|MB_ICONINFORMATION);
	delete [] comlist;*/
}


//9节点网络文件生成，代码如下
void CComplexNetView::On9nodes()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("九节点网络生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（九节点网络文件的生成代码，可作为其他网络生成的参考）
	    long int i;
	    long int fr,to;
	    char a[256];
		char temp[256];	
		int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
    	strcpy_s(a,"*Nine Nodes Network\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",9);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
		double coordinate[]={0.194,0.417,0.194,0.667,0.325,0.597,0.490,0.597,0.647,0.597,
						0.386,0.417,0.586,0.417,0.780,0.417,0.780,0.667};
	    for(i=0;i<9;i++)
	    { 
	       sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, coordinate[2*i],  coordinate[2*i+1],0.5);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
        strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));
	    int nodetonode[]={1,2,1,3,2,3,3,4,3,6,4,5,5,7,5,8,5,9,6,7,8,9};
	    for(i=0;i<11;i++)
	    { 
	       fr=nodetonode[2*i];
	       to=nodetonode[2*i+1];
	       sprintf_s(temp,"%d %d 1\n",fr,to);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
	    _close(fh);		

		//NetworkFun.dll 存储Dot、Png文件
		CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//10节点网络文件生成,代码如下
void CComplexNetView::On10nodes()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("十节点网络生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（十节点网络文件的生成代码，可作为其他网络生成的参考）
	    long int i;
	    long int fr,to;
	    char a[256];
		char temp[256];		
	    int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
		strcpy_s(a,"*Ten Nodes Network\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",10);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
	    double coordinate[]={0.440,0.770,0.564,0.77,0.630,0.588,0.761,0.445,0.503,0.447,
						0.37,0.588,0.227,0.588,0.227,0.382,0.37,0.382,0.503,0.207};
	    for(i=0;i<10;i++)
	    {
	      sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, coordinate[2*i],  coordinate[2*i+1],0.5);
	      strcpy_s(a,temp);
		  _write(fh,a,strlen(a));
	    }
	    strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));
	    int nodetonode[]={1,2,1,3,1,5,1,6,1,7,2,3,2,5,2,6,2,7,3,4,3,5,3,6,4,5,4,6,5,6,5,
					  9,5,10,6,7,6,8,6,9,7,8,7,9,8,9,9,10};
	    for(i=0;i<24;i++)
	    {
	      fr=nodetonode[2*i];
	      to=nodetonode[2*i+1];
	      sprintf_s(temp,"%d %d 1\n",fr,to);
	      strcpy_s(a,temp);
		  _write(fh,a,strlen(a));
	    }
	    _close(fh);
		//NetworkFun.dll 存储Dot、Png文件
        CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//13节点网络1文件生成,代码如下
void CComplexNetView::On13nodes()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("十三节点网络（1）生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（十三节点网络文件的生成代码，可作为其他网络生成的参考）
        long int i;
	    long int fr,to;
	    char a[256];
	    char temp[256];		
	    int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
		strcpy_s(a,"*Thirteen Nodes Network One\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",13);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
		double coordinate[]={0.467,0.220,0.409,0.340,0.529,0.340,0.467,0.463,0.467,0.582,
						0.395,0.678,0.290,0.695,0.366,0.820,0.246,0.820,0.546,0.678,
						0.636,0.695,0.573,0.820,0.695,0.820};
	    for(i=0;i<13;i++)
	    {
	       sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, coordinate[2*i],  coordinate[2*i+1],0.5);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));
	    int nodetonode[]={1,2,1,3,2,3,2,4,3,4,4,5,5,6,5,10,6,7,6,8,7,8,7,9,8,9,10,11,10,
					  12,11,12,11,13,12,13};
	    for(i=0;i<18;i++)
	    {
	       fr=nodetonode[2*i];
	       to=nodetonode[2*i+1];
	       sprintf_s(temp,"%d %d 1\n",fr,to);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
		}
	    _close(fh);
		//NetworkFun.dll 存储Dot、Png文件
		CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//13节点网络2文件生成,代码如下
void CComplexNetView::On13nodestwo()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("十三节点网络（2）生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（十三节点网络文件的生成代码，可作为其他网络生成的参考）
	    long int i;
	    long int fr,to;
	    char a[256];
	    char temp[256];		
	    int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
		strcpy_s(a,"*Thirteen Nodes Network Two\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",13);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
	    double coordinate[]={0.425,0.250,0.277,0.375,0.500,0.375,0.353,0.503,0.191,0.587,
						0.501,0.603,0.276,0.698,0.407,0.765,0.500,0.867,0.627,0.765,
						0.656,0.502,0.783,0.680,0.814,0.373};
	    for(i=0;i<13;i++)
	    {
	       sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, coordinate[2*i],  coordinate[2*i+1],0.5);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));
	    int nodetonode[]={1,2,1,3,2,3,2,4,3,4,3,6,3,11,4,5,4,6,4,8,5,7,6,8,6,10,6,11,7,8,
					  8,9,8,10,9,10,10,11,10,12,11,12,11,13};
 	    for(i=0;i<22;i++)
	    {
	       fr=nodetonode[2*i];
	       to=nodetonode[2*i+1];
	       sprintf_s(temp,"%d %d 1\n",fr,to);
           strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		_close(fh);
		//NetworkFun.dll 存储Dot、Png文件
        CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//性关系网络文件生成,代码如下
void CComplexNetView::OnSexnetwork()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("性关系网络生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（性关系网络文件的生成代码，可作为其他网络生成的参考）
	    long int i;
	    long int fr,to;
	    char a[256];
	    char temp[256];		
	    int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
		strcpy_s(a,"*Sex Network\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",40);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
	 	double coordinate[]={0.156,0.305,0.133,0.432,0.095,0.528,0.121,0.622,0.185,0.493,
						0.233,0.395,0.184,0.747,0.234,0.655,0.287,0.745,0.379,0.750,
						0.270,0.565,0.356,0.652,0.456,0.783,0.456,0.677,0.323,0.493,
						0.430,0.568,0.384,0.457,0.585,0.783,0.627,0.692,0.533,0.712,
						0.456,0.458,0.524,0.445,0.554,0.543,0.564,0.642,0.482,0.323,
						0.660,0.475,0.627,0.555,0.686,0.590,0.741,0.690,0.687,0.778,
						0.769,0.508,0.810,0.623,0.806,0.728,0.893,0.655,0.866,0.510,
						0.810,0.388,0.866,0.305,0.686,0.387,0.757,0.305,0.591,0.327};
	    for(i=0;i<40;i++)
	    {
	      sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, coordinate[2*i],  coordinate[2*i+1],0.5);
	      strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));	
	    int nodetonode[]={1,2,2,5,3,5,4,5,5,6,5,11,7,8,8,9,8,11,9,10,11,16,12,16,13,14,14,
					  16,15,16,16,17,16,20,16,21,16,22,18,20,19,20,19,28,22,23,22,25,
					  22,26,23,24,26,27,26,28,26,31,26,38,28,29,29,30,31,32,31,36,32,
					  33,32,34,33,34,34,35,36,37,38,39,38,40};
	    for(i=0;i<41;i++)
	    {
	       fr=nodetonode[2*i];
	       to=nodetonode[2*i+1];
	       sprintf_s(temp,"%d %d 1\n",fr,to);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		_close(fh);
		//NetworkFun.dll 存储Dot、Png文件
		CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//空手道网络文件生成,代码如下
void CComplexNetView::OnKarate()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDlgExampleNets egnet;
	egnet.path=pDoc->CurrentWorkPath;
	egnet.title=_T("空手道俱乐部网络生成");
	if(egnet.DoModal()==IDOK)
	{
		//这里给出生成网络文件的例子（空手道网络文件的生成代码，可作为其他网络生成的参考）
	    long int i;
	    long int fr,to;
	    char a[256];
		char temp[256];		
	    int fh;
	    _sopen_s(&fh,egnet.path,_O_CREAT|_O_TEXT|_O_TRUNC|_O_WRONLY,_SH_DENYNO,_S_IREAD|_S_IWRITE);
	    if(fh==-1)
		{
		   MessageBox("Cannot create the net file");
		   return;
		}
		strcpy_s(a,"Katate Network\n");
	    _write(fh,a,strlen(a));
	    sprintf_s(temp," %d\n",34);
	    strcpy_s(a,"*Vertices");
	    strcat_s(a,temp);
	    _write(fh,a,strlen(a));
	    double karcoordinate[]={0.581,0.562,0.506,0.543,0.403,0.552,0.561,0.445,0.675,0.452,
						   0.729,0.473,0.735,0.570,0.501,0.698,0.401,0.475,0.369,0.293,
						   0.687,0.630,0.639,0.812,0.647,0.340,0.456,0.420,0.234,0.282,
						   0.173,0.357,0.860,0.530,0.563,0.795,0.149,0.560,0.483,0.363,
						   0.133,0.485,0.637,0.727,0.289,0.257,0.250,0.615,0.336,0.838,
						   0.291,0.790,0.137,0.283,0.295,0.653,0.323,0.588,0.165,0.452,
						   0.367,0.372,0.366,0.633,0.270,0.485,0.287,0.435};
	    for(i=0;i<34;i++)
	    {
	       sprintf_s(temp,"%d %d %f %f %f\n", i+1,i+1, karcoordinate[2*i],  karcoordinate[2*i+1],0.5);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
		strcpy_s(a,"*Edges\n");
	    _write(fh,a,strlen(a));
	    int nodetonode[]={2,1,3,1,3,2,4,1,4,2,4,3,5,1,6,1,7,1,7,5,7,6,8,1,8,2,8,3,8,4,9,1,
					  9,3,10,3,11,1,11,5,11,6,12,1,13,1,13,4,14,1,14,2,14,3,14,4,17,6,
					  17,7,18,1,18,2,20,1,20,2,22,1,22,2,26,24,26,25,28,3,28,24,28,25,
					  29,3,30,24,30,27,31,2,31,9,32,1,32,25,32,26,32,29,33,3,33,9,33,
					  15,33,16,33,19,33,21,33,23,33,24,33,30,33,31,33,32,34,9,34,10,
					  34,14,34,15,34,16,34,19,34,20,34,21,34,23,34,24,34,27,34,28,34,
					  29,34,30,34,31,34,32,34,33};
	    for(i=0;i<78;i++)
	    { 
	       fr=nodetonode[2*i];
	       to=nodetonode[2*i+1];
	       sprintf_s(temp,"%d %d 1\n",fr,to);
	       strcpy_s(a,temp);
		   _write(fh,a,strlen(a));
	    }
	    _close(fh);
		//NetworkFun.dll 存储Dot、Png文件
		CString tempstr;
		UNetwork<>::pNetwork tempnet;
		tempstr=egnet.path.Left(egnet.path.GetLength()-4);
		pDoc->ReadUNetworkFromNetFile(tempnet,(char *)egnet.path.GetString());
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//规则网络生成，函数GenKNearestNetwork,writeToNetFile来自NetworkFun.dll
void CComplexNetView::OnNewRegular()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CRegularDlg crd;
	crd.path=pDoc->CurrentWorkPath;
	if(crd.DoModal()==IDOK)
	{
	   //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
	   UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenKNearestNetwork(crd.m_reg_nodes,crd.m_reg_neighbors)));
	   pDoc->WriteToNetFile((char *)crd.path.GetString(),tempnet);
	   CString tempstr;
	   tempstr=crd.path.Left(crd.path.GetLength()-4);
	   pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}


}


//随机网络生成，函数RandomNetworkGenerate来自NetworkFun.dll
void CComplexNetView::OnNetRandom()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CRandomDlg crd;
	crd.path=pDoc->CurrentWorkPath;
	if(crd.DoModal()==IDOK)
	{
	   //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
	   double prob;
	   prob=2.0*(double)crd.m_ran_edges/(crd.m_ran_nodes*(crd.m_ran_nodes-1.0));
	   UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenRandomNetwork(crd.m_ran_nodes,prob)));
	   pDoc->WriteToNetFile((char *)crd.path.GetString(),tempnet);
	   CString tempstr;
	   tempstr=crd.path.Left(crd.path.GetLength()-4);
	   pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//随机均匀迭代树网络生成，函数GenUniformRecursiveTree来自NetworkFun.dll
void CComplexNetView::OnNetUrt()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("随机均匀迭代树网络生成");
	cdd.flag=1;
	if(cdd.DoModal()==IDOK)
	{
	    //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenUniformRecursiveTree(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//WS小世界网络生成，函数GenSmallWorldNetworkByWS来自NetworkFun.dll
void CComplexNetView::OnNetWssmall()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CWSSmallDlg cwd;
	cwd.path=pDoc->CurrentWorkPath;
	if(cwd.DoModal()==IDOK)
	{
	  //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
   	  UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenSmallWorldNetworkByWS(cwd.m_wssmall_nodes,cwd.m_wssmall_neighbors,cwd.m_wssmall_prob)));
	  pDoc->WriteToNetFile((char *)cwd.path.GetString(),tempnet);
	  CString tempstr;
	  tempstr=cwd.path.Left(cwd.path.GetLength()-4);
	  pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}


//NW小世界网络生成，函数GenSmallWorldNetworkByNW来自NetworkFun.dll
void CComplexNetView::OnNetNWSmall()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CNWSmallDlg cnd;
	cnd.path=pDoc->CurrentWorkPath;
	if(cnd.DoModal()==IDOK)
	{
	  //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
   	  UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenSmallWorldNetworkByNW(cnd.m_nwsmall_nodes,cnd.m_nwsmall_neighbors,cnd.m_nwsmall_prob)));
	  pDoc->WriteToNetFile((char *)cnd.path.GetString(),tempnet);
	  CString tempstr;
	  tempstr=cnd.path.Left(cnd.path.GetLength()-4);
	  pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}


//BA无标度网络生成，函数GenScaleFreeNetwork来自NetworkFun.dll
void CComplexNetView::OnNetBA()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CBAModelDlg cbd;
	cbd.path=pDoc->CurrentWorkPath;
	if(cbd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenScaleFreeNetwork(cbd.m_ba_nodes,cbd.m_ba_onodes)));
	    pDoc->WriteToNetFile((char *)cbd.path.GetString(),tempnet);
		CString tempstr;
	    tempstr=cbd.path.Left(cbd.path.GetLength()-4);
	    pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//边迭代确定性小世界网络生成，函数GenSmallWorldByEdgeIteratione来自NetworkFun.dll
void CComplexNetView::OnDeterminswei()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("基于边迭代的确定性小世界网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
	    //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenSmallWorldByEdgeIteration(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}

//确定性迭代树网络生成，函数GenDURT来自NetworkFun.dll
void CComplexNetView::OnDeterminUrt()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("确定性均匀迭代树网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
	    //NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenDURT(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}


//确定性小世界迭代树网络生成，函数GenSmallWorldNetworkFromDURT来自NetworkFun.dll
//陆哲明和郭世泽的算法
//Zhe-Ming Lu, and Shi-Ze Guo(2012). A small-world network derived from the deterministic uniform recursive tree. Physica A. 391(1-2), 87-92.
void CComplexNetView::OnDeterminSwurt()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("确定性均匀迭代树小世界网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		////NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenSmallWorldNetworkFromDURT(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}

//确定性风车小世界网络生成，函数GenPinWheelShapedSW来自NetworkFun.dll 
//郭世泽、李宠、陆哲明和乔玉龙的算法
//Shi-Ze Guo, Chong Li, Zhe-Ming Lu and Yu-Long Qiao (2012). A Pinwheel-Shaped Determinstic Small-World Network. ICIC Express Letters. 6(7):1809-1814.
void CComplexNetView::OnDeterminPinwheel()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("风车确定性小世界网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenPinWheelShapedSW(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}

//确定性漩涡网络生成，函数GenSwirlShapedNetwork来自NetworkFun.dll 
//李宠、郭世泽、陆哲明和乔玉龙算法
// Chong Li, Shi-Ze Guo, Zhe-Ming Lu and Yu-Long Qiao(2011): A Swirl-shaped Deterministic Network with High Clustering Coefficient. Information Technology Journal. 10(10):1994-1998.
void CComplexNetView::OnDeterminSwirl()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("漩涡确定性网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenSwirlShapedNetwork(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}


//树状无标度小世界确定性网络生成，函数GenTreeStructuredSFSW来自NetworkFun.dll
//Zhe-Ming Lu, Yu-Xin Su, and Shi-Ze Guo(2013). Deterministic scale-free small-world network of arbitrary order. Physica A.
void CComplexNetView::OnDeterminTssfsw()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("树状无标度小世界确定性网络生成");
	cdd.flag=1;
	if(cdd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenTreeStructuredSFSW(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}


//树状小世界确定性网络生成，函数GenTreeStructuredSW来自NetworkFun.dll
//郭世泽、陆哲明等算法
//Shi-Ze Guo, Zhe-Ming Lu, Guang-Yu Kang, Zhe Chen and Hao Luo (2012): A tree-structured deterministic small-world network. IEICE Transactions on Information and Systems. E95D(5): pp. 1536-1538.
void CComplexNetView::OnDeterminTs()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("树状小世界确定性网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenTreeStructuredSW(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}

//改进的边迭代小世界确定性网络生成，函数GenTriangleExtendedDSWN来自NetworkFun.dll 
//陆哲明、李新峰等算法，论文待投
void CComplexNetView::OnDeterminMei()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("改进的边迭代小世界确定性网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenTriangleExtendedDSWN(cdd.m_determin_iterations)));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}

//盒计数法聚类，函数RenormalizeByBoxCounting来自NetworkFun.dll 
void CComplexNetView::OnBoxCounting()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox("Please open a network file first","Warning!", MB_OK|MB_ICONEXCLAMATION);
		return;
	}
	pDoc->unet.reset(new UNetwork<>(pDoc->RenormalizeByBoxCounting(pDoc->unet->GetTopology(),3)));
	Invalidate(FALSE);
}


void CComplexNetView::OnGroupSpectrum()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	char temp[256];
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	sprintf_s(temp,"%s     %.6f\r\n","Average Degree=",pDoc->RenormalizeBySpectralBisection(pDoc->unet->GetTopology(),3));
    MessageBox(temp);

}


void CComplexNetView::OnGroupGn()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	char temp[256];
	if(pDoc->NetTxtFileOpened==FALSE)
	{
		MessageBox(_T("Please open a network file first"));
		return;
	}
	vector< vector<size_t>> CmtyV;
	sprintf_s(temp,"%s     %.6f\r\n","Average Degree=",pDoc->CommunityGirvanNewman(pDoc->unet->GetTopology(),CmtyV));
    MessageBox(temp);
}



//生成社团网络，函数GenCommunityNetwork来自NetworkFun.dll
void CComplexNetView::OnGroupnetSimple()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CGroupNetDlg cgnd;
	cgnd.path=pDoc->CurrentWorkPath;
	if(cgnd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UNetwork<>::pNetwork tempnet(new UNetwork<>(pDoc->GenCommunityNetwork(cgnd.m_groupnet_nodes,cgnd.m_groupnet_groups,cgnd.m_groupnet_proba,cgnd.m_groupnet_probb)));
	    pDoc->WriteToNetFile((char *)cgnd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cgnd.path.Left(cgnd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}	
}

//目前计算特征值函数还不好使，等待更新
void CComplexNetView::OnSyncLamda2()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	//NetworkFun.dll
	//double a,b;
	//目前计算特征值函数还不好使，等待更新
	MessageBox("目前计算特征值函数还不好使，等待更新");
	//pDoc->GetLambda2AndRatio(a,b,pDoc->unet->GetTopology());
	
}

//文件保存Net,dot,Png三种格式
void CComplexNetView::OnFileSave()
{
	// TODO: 在此添加命令处理程序代码
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	//NetworkFun.dll Net,dot,Png文件存储
	pDoc->WriteToNetFile((char *)pDoc->NetFileName.GetString(),pDoc->unet);
	CString tempstr;
	tempstr=pDoc->NetFileName.Left(pDoc->NetFileName.GetLength()-4);
	pDoc->DrawCircleForm(pDoc->unet->GetTopology(),tempstr.GetString());
	Invalidate(FALSE);
}





//文件另存Net,dot,Png三种格式
void CComplexNetView::OnFileSaveAs()
{
	// TODO: 在此添加命令处理程序代码
   CComplexNetDoc* pDoc = GetDocument();
   ASSERT_VALID(pDoc);
   CNetFileSaveAs cnfsa;
   cnfsa.path=pDoc->CurrentWorkPath;
   if(cnfsa.DoModal()==IDOK)
   {
	  //NetworkFun.dll Net,dot,Png文件存储
	  pDoc->WriteToNetFile((char *)cnfsa.path.GetString(),pDoc->unet);
	  CString tempstr;
	  tempstr=cnfsa.path.Left(cnfsa.path.GetLength()-4);
	  pDoc->DrawCircleForm(pDoc->unet->GetTopology(),tempstr.GetString());
	  
	  //改变显示的文件标题
	  pDoc->NetFileName=cnfsa.path;
	  pDoc->NetFileTitle=cnfsa.path.Right(cnfsa.path.GetLength()-pDoc->NetFileName.ReverseFind('\\')-1);
	  pDoc->SetTitle(pDoc->NetFileTitle);
	  Invalidate(FALSE);
   }
}








void CComplexNetView::OnNetRelation()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("关系网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UGraph::pGraph graph = pDoc->GenRelationNetwork(cdd.m_determin_iterations);
		UNetwork<>::pNetwork tempnet(new UNetwork<>(graph));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}

}


void CComplexNetView::OnNetPmnetwork()
{
	// TODO: Add your command handler code here
	CComplexNetDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CDeterminDlg cdd;
	cdd.path=pDoc->CurrentWorkPath;
	cdd.title=_T("兴趣偏好记忆网络生成");
	cdd.flag=0;
	if(cdd.DoModal()==IDOK)
	{
		//NetworkFun.dll 生成网络并存储Net、Dot、Png文件
		UGraph::pGraph graph = pDoc->GenPreferenceMemoryNetwork(cdd.m_determin_iterations);
		UNetwork<>::pNetwork tempnet(new UNetwork<>(graph));
	    pDoc->WriteToNetFile((char *)cdd.path.GetString(),tempnet);
		CString tempstr;
		tempstr=cdd.path.Left(cdd.path.GetLength()-4);
		pDoc->DrawCircleForm(tempnet->GetTopology(),tempstr.GetString());
	}
}





