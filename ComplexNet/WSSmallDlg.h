#pragma once


// CWSSmallDlg �Ի���

class CWSSmallDlg : public CDialog
{
	DECLARE_DYNAMIC(CWSSmallDlg)

public:
	CWSSmallDlg(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CWSSmallDlg();
    CString path;
// �Ի�������
	enum { IDD = IDD_DIALOG_WSSMALL };
    CString	m_wssmall_name;
	long	m_wssmall_neighbors;
	long	m_wssmall_nodes;
	double	m_wssmall_prob;
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
    virtual void OnOK();
	DECLARE_MESSAGE_MAP()
};


