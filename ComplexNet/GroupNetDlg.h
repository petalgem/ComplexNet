#pragma once


// CGroupNetDlg �Ի���

class CGroupNetDlg : public CDialog
{
	DECLARE_DYNAMIC(CGroupNetDlg)

public:
	CGroupNetDlg(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CGroupNetDlg();
	CString path;
// �Ի�������
	enum { IDD = IDD_DIALOG_GROUPNET };
	CString	m_groupnet_name;
	long	m_groupnet_groups;
	long	m_groupnet_nodes;
	double	m_groupnet_proba;
	double	m_groupnet_probb;
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
	virtual void OnOK();
	DECLARE_MESSAGE_MAP()
};
