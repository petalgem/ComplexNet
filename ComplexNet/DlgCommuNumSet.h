#pragma once


// CDlgCommuNumSet �Ի���

class CDlgCommuNumSet : public CDialogEx
{
	DECLARE_DYNAMIC(CDlgCommuNumSet)

public:
	CDlgCommuNumSet(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~CDlgCommuNumSet();

// �Ի�������
	enum { IDD = IDD_DIALOG_KLDATA };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

	DECLARE_MESSAGE_MAP()
public:
	int m_Commu_One_Num;
	int m_Commu_Two_Num;
	afx_msg void OnBnClickedOk();
};
