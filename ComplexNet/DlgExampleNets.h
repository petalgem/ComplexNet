#pragma once



// CDlgExampleNets �Ի���

class CDlgExampleNets : public CDialog
{
	DECLARE_DYNAMIC(CDlgExampleNets)

public:
	CDlgExampleNets(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CDlgExampleNets();
	virtual BOOL OnInitDialog();
// �Ի�������
	enum { IDD = IDD_DIALOG_EXAMPLENETS };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
	CString m_netfile;
	CString path;
	CString title;
};
