#pragma once


// CNetFileSaveAs �Ի���

class CNetFileSaveAs : public CDialog
{
	DECLARE_DYNAMIC(CNetFileSaveAs)

public:
	CNetFileSaveAs(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CNetFileSaveAs();
    CString path;
// �Ի�������
	enum { IDD = IDD_DIALOG_SAVEFILE };
	CString	m_savefile_name;
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��
	virtual void OnOK();
	DECLARE_MESSAGE_MAP()
};
