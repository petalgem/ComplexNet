
// ComplexNetView.h : CComplexNetView 类的接口
//

#pragma once


class CComplexNetView : public CScrollView
{
protected: // 仅从序列化创建
	CComplexNetView();
	DECLARE_DYNCREATE(CComplexNetView)

// 特性
public:
	CComplexNetDoc* GetDocument() const;

// 操作
public:

// 重写
public:
	virtual void OnDraw(CDC* pDC);  // 重写以绘制该视图
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void OnInitialUpdate(); // 构造后第一次调用
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// 实现
public:
	virtual ~CComplexNetView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 生成的消息映射函数
protected:
	afx_msg void OnFilePrintPreview();
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnContextMenu(CWnd* pWnd, CPoint point);
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void On9nodes();
	afx_msg void OnFeatureDegree();
	afx_msg void OnFeatureDegreedis();
	afx_msg void OnFeatureDegreecorr();
	afx_msg void OnFeatureCluster();
	afx_msg void OnFeatureAvecluster();
	afx_msg void OnFeatureClusterdis();
	afx_msg void OnFeatureClusterdegree();
	afx_msg void OnFeatureApl();
	afx_msg void OnFeaturePathdis();
	afx_msg void OnAppEigenvector();
	afx_msg void OnCloseness();
	afx_msg void OnAppFlowBetw();
	afx_msg void OnNomination();
	afx_msg void OnKernighanLin();
	afx_msg void On13nodes();
	afx_msg void On13nodestwo();
	afx_msg void On10nodes();
	afx_msg void OnSexnetwork();
	afx_msg void OnKarate();
	afx_msg void OnNewRegular();
	afx_msg void OnNetRandom();
	afx_msg void OnNetWssmall();
	afx_msg void OnNetNWSmall();
	afx_msg void OnNetBA();
	afx_msg void OnNetLiuLai();
	afx_msg void OnDeterminswei();
	afx_msg void OnDeterminUrt();
	afx_msg void OnDeterminSwurt();
	afx_msg void OnDeterminPinwheel();
	afx_msg void OnDeterminSwirl();
	afx_msg void OnDeterminTssfsw();
	afx_msg void OnDeterminTs();
	afx_msg void OnDeterminMei();
	afx_msg void OnBoxCounting();
	afx_msg void OnGroupnetSimple();
	afx_msg void OnSyncLamda2();
	afx_msg void OnFileSave();
	afx_msg void OnFileSaveAs();
	afx_msg void OnFeatureAvedegree();
	afx_msg void OnNetUrt();
	afx_msg void OnNetRelation();
	afx_msg void OnNetPmnetwork();
};

#ifndef _DEBUG  // ComplexNetView.cpp 中的调试版本
inline CComplexNetDoc* CComplexNetView::GetDocument() const
   { return reinterpret_cast<CComplexNetDoc*>(m_pDocument); }
#endif

