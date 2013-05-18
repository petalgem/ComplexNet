
// ComplexNetDoc.cpp : CComplexNetDoc ���ʵ��
//

#include "stdafx.h"
// SHARED_HANDLERS ������ʵ��Ԥ��������ͼ������ɸѡ�������
// ATL ��Ŀ�н��ж��壬�����������Ŀ�����ĵ����롣
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


// CComplexNetDoc ����/����

CComplexNetDoc::CComplexNetDoc()
{
	// TODO: �ڴ����һ���Թ������
   NetTxtFileOpened=FALSE;
   
   //��õ�ǰ����Ŀ¼
   GetCurrentDirectory(_MAX_PATH,CurrentWorkPath);

   //��ö�̬���ӿ�ʵ��
   hInstance = LoadLibrary("NetworkFun.dll");
   
   //��ö�̬���ӿ⺯���ӿ�+++++++++++Ŀǰ֧����������+++++++++++++++++++++++++++++++++++++++��ʼ
   //����UNetwork�ṹ�洢����dot�ļ�
   WriteToDotFile=(writeToDotFile)GetProcAddress(hInstance,"WriteToDotFile");
   //����UNetwork�ṹ�洢����bmp�ļ�
   DrawCircleForm=(drawCircleForm)GetProcAddress(hInstance,"DrawCircleForm");
   //����UNetwork�ṹ�洢����net�ļ�
   WriteToNetFile=(writeToNetFile)GetProcAddress(hInstance,"WriteToNetFile");
   //��ȡnet�ļ���UNetwork�ṹ
   ReadUNetworkFromNetFile=(readUNetworkFromNetFile)GetProcAddress(hInstance,"ReadUNetworkFromNetFile");
   //����K���������������ɵĺ���
   GenKNearestNetwork=(genKNearestNetwork)GetProcAddress(hInstance,"GenKNearestNetwork");
   //����������ɵĺ���
   GenRandomNetwork=(genRandomNetwork)GetProcAddress(hInstance,"GenRandomNetwork");
   //WSС�����������ɵĺ���
   GenSmallWorldNetworkByWS=(genSmallWorldNetworkByWS)GetProcAddress(hInstance,"GenSmallWorldNetworkByWS");
   //NWС�����������ɵĺ���
   GenSmallWorldNetworkByNW=(genSmallWorldNetworkByNW)GetProcAddress(hInstance,"GenSmallWorldNetworkByNW");
   //BA�ޱ���������ɵĺ���
   GenScaleFreeNetwork=(genScaleFreeNetwork)GetProcAddress(hInstance,"GenScaleFreeNetwork");
   //��״С����ȷ������������
   GenTreeStructuredSW=(genTreeStructuredSW)GetProcAddress(hInstance,"GenTreeStructuredSW");
   //��״�ޱ��С����ȷ������������
   GenTreeStructuredSFSW=(genTreeStructuredSFSW)GetProcAddress(hInstance,"GenTreeStructuredSFSW");


   GenRelationNetwork=(genRelationNetwork)GetProcAddress(hInstance,"GenRelationNetwork");
   GenPreferenceMemoryNetwork=(genPreferenceMemoryNetwork)GetProcAddress(hInstance,"GenPreferenceMemoryNetwork");


   //�ߵ���С�����������ɺ���
   GenSmallWorldByEdgeIteration=(genSmallWorldByEdgeIteration)GetProcAddress(hInstance,"GenSmallWorldByEdgeIteration");
   //���ȵ������������ɺ���
   GenUniformRecursiveTree=(genUniformRecursiveTree)GetProcAddress(hInstance,"GenUniformRecursiveTree");
   //ȷ���Ծ��ȵ������������ɺ���
   GenDURT=(genDURT)GetProcAddress(hInstance,"GenDURT");
   //���ȵ�����С�����������ɺ���
   GenSmallWorldNetworkFromDURT=(genSmallWorldNetworkFromDURT)GetProcAddress(hInstance,"GenSmallWorldNetworkFromDURT");
   //�Ľ��ߵ���С�����������ɺ���
   GenTriangleExtendedDSWN=(genTriangleExtendedDSWN)GetProcAddress(hInstance,"GenTriangleExtendedDSWN");
   GenSwirlShapedNetwork=(genSwirlShapedNetwork)GetProcAddress(hInstance,"GenSwirlShapedNetwork");
    GenPinWheelShapedSW=(genPinWheelShapedSW)GetProcAddress(hInstance,"GenPinWheelShapedSW");
   //������������
   GenCommunityNetwork=(genCommunityNetwork)GetProcAddress(hInstance,"GenCommunityNetwork");
   //���ݶȷֲ��������磨��ʱ��Ч��
   GenNetworkFromDegreeDistribution=(genNetworkFromDegreeDistribution)GetProcAddress(hInstance,"GenNetworkFromDegreeDistribution");
   IsDegreeListGraphical=(isDegreeListGraphical)GetProcAddress(hInstance,"IsDegreeListGraphical");
   //�м���������
   RenormalizeByBoxCounting=(renormalizeByBoxCounting)GetProcAddress(hInstance,"RenormalizeByBoxCounting");
   
   RenormalizeBySpectralBisection=(renormalizeBySpectralBisection)GetProcAddress(hInstance,"RenormalizeBySpectralBisection");
   
   //����ƽ����
   ComputeAverageDegree=(computeAverageDegree)GetProcAddress(hInstance,"ComputeAverageDegree");
   //������
   ComputeSpectrum=(computeSpectrum)GetProcAddress(hInstance,"ComputeSpectrum");
   //����ͬ������
   GetLambda2AndRatio=(getLambda2AndRatio)GetProcAddress(hInstance,"GetLambda2AndRatio");
   //����ȷֲ�
   GetDegreeDistribution=(getDegreeDistribution)GetProcAddress(hInstance,"GetDegreeDistribution");
   //����ȶ������
   GetPSDegreeCorrelation=(getPSDegreeCorrelation)GetProcAddress(hInstance,"GetPSDegreeCorrelation");
   //����Pearson�ȶ����ϵ��
   GetPearsonCorrCoeff=(getPearsonCorrCoeff)GetProcAddress(hInstance,"GetPearsonCorrCoeff");
    
   //�������ܶ�
   ComputeSpectralDensity=(computeSpectralDensity)GetProcAddress(hInstance,"ComputeSpectralDensity");
   //����ȷֲ���
   GetEntropyOfDegreeDist=(getEntropyOfDegreeDist)GetProcAddress(hInstance,"GetEntropyOfDegreeDist");
   //������̾���
   GetShortestDistance=(getShortestDistance)GetProcAddress(hInstance,"GetShortestDistance");
   //�����̾������
   GetGeodesicMatrix=(getGeodesicMatrix)GetProcAddress(hInstance,"GetGeodesicMatrix");
   //����ֱ��
   GetDiameter=(getDiameter)GetProcAddress(hInstance,"GetDiameter");
   //����ֱ����ƽ����̾���
   GetDiameterAndAverageDistance=(getDiameterAndAverageDistance)GetProcAddress(hInstance,"GetDiameterAndAverageDistance");
   //�������ֲ�
   GetShortestDistanceDistribution=(getShortestDistanceDistribution)GetProcAddress(hInstance,"GetShortestDistanceDistribution");
   //����ƽ������
   GetAverageDistanceByDjikstra=(getAverageDistanceByDjikstra)GetProcAddress(hInstance,"GetAverageDistanceByDjikstra");
   //����ƽ������
   GetAverageDistance=(getAverageDistance)GetProcAddress(hInstance,"GetAverageDistance");
   //��������ȫ��Ч��
   GetGlobalEfficiency=(getGlobalEfficiency)GetProcAddress(hInstance,"GetGlobalEfficiency");
   //�������������
   GetVulnerability=(getVulnerability)GetProcAddress(hInstance,"GetVulnerability");
   GetTransitivity=(getTransitivity)GetProcAddress(hInstance,"GetTransitivity");
   //���㼯��ϵ��
   GetClusteringCoeff=(getClusteringCoeff)GetProcAddress(hInstance,"GetClusteringCoeff");
   //���㼯��ϵ���ֲ�
   GetClusteringCoeffDist=(getClusteringCoeffDist)GetProcAddress(hInstance,"GetClusteringCoeffDist");
   //�����-�������
   GetClusteringDegreeCorre=(getClusteringDegreeCorre)GetProcAddress(hInstance,"GetClusteringDegreeCorre");
   
   //���º��������У��ȴ�����
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
   //��ö�̬���ӿ⺯���ӿ�++++++++++Ŀǰ֧����������+++++++++++++++++++++++++++++++++++++++��ʼ

   //����NetFiles��ResultsĿ¼++++++++++++��ʼ
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
   //����NetFiles��ResultsĿ¼++++++++++++����

   //���BMPͼ����Ϣͷ
   OpenStandardBMPHeader();
}

CComplexNetDoc::~CComplexNetDoc()
{
}

BOOL CComplexNetDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: �ڴ�������³�ʼ������
	// (SDI �ĵ������ø��ĵ�)
	if(Success==FALSE)return FALSE;
	return TRUE;
}

//��ȡBMP��Ϣͷ��Ϊ�˴洢��������BMPͼ��
void CComplexNetDoc::OpenStandardBMPHeader()
{
	Success=TRUE;
    long int fh;
    LPBITMAPFILEHEADER bitfile;
    LPBITMAPINFOHEADER bitinfo;
    //open the standard file
	//��ȡ256�Ҷ�ͼ��BMP��Ϣͷ
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

	//��ȡ��ɫͼ��BMP��Ϣͷ
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
// CComplexNetDoc ���л�
void CComplexNetDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: �ڴ���Ӵ洢����
		//�洢��ר�ź�������OnFileSave()��OnFileSaveAs()
	}
	else
	{
		// TODO: �ڴ���Ӽ��ش���
		NetTxtFileOpened=FALSE;
		//Get the file pointer
		CFile *fp;
		//char filenm[_MAX_PATH];//�鵵�ļ�·������
		fp=ar.GetFile();//��ô˹鵵�ļ���CFile����ָ�� 
	    ULONGLONG flength;
		flength=fp->GetLength();//����ļ���С
		if(flength<=0)return;
		NetFileName=fp->GetFilePath();//����ļ�·��
       
		//��ȡ�����ļ���unet�ṹ��Ŀǰֻ֧����������
		ReadUNetworkFromNetFile(unet,(char *)NetFileName.GetString());
		if(unet->GetTopology()->GetNumberOfNodes()==0)return;
			
		NetTxtFileOpened=TRUE;
		NetFileTitle=fp->GetFileTitle();
	}
}

#ifdef SHARED_HANDLERS

// ����ͼ��֧��
void CComplexNetDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// �޸Ĵ˴����Ի����ĵ�����
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

// ������������֧��
void CComplexNetDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// ���ĵ����������������ݡ�
	// ���ݲ���Ӧ�ɡ�;���ָ�

	// ����:  strSearchContent = _T("point;rectangle;circle;ole object;")��
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

// CComplexNetDoc ���

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

// CComplexNetDoc ����
void CComplexNetDoc::SetTitle(LPCTSTR lpszTitle)
{
	// TODO: �ڴ����ר�ô����/����û���

	CDocument::SetTitle(lpszTitle);
	CString title;
	if(NetFileTitle=="")
	  lpszTitle="û�������ļ���";
	else
	{
	  title=lpszTitle;
	  title+=" �����ļ�";
	  lpszTitle=(LPCTSTR)title;
	}
	CDocument::SetTitle(lpszTitle);

}
