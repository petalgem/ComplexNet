#pragma once


// CDeterminDlg �Ի���

class CDeterminDlg : public CDialog
{
	DECLARE_DYNAMIC(CDeterminDlg)

public:
	CDeterminDlg(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CDeterminDlg();
	virtual BOOL OnInitDialog();
	CString path;
	CString title;
	int flag;
// �Ի�������
	enum { IDD = IDD_DIALOG_DETERMIN };
    long m_determin_iterations;
	CString	m_determin_name;
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
	virtual void OnOK();
	
	DECLARE_MESSAGE_MAP()
};
