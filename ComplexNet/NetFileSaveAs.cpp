// NetFileSaveAs.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "ComplexNet.h"
#include "NetFileSaveAs.h"
#include "afxdialogex.h"


// CNetFileSaveAs �Ի���

IMPLEMENT_DYNAMIC(CNetFileSaveAs, CDialog)

CNetFileSaveAs::CNetFileSaveAs(CWnd* pParent /*=NULL*/)
	: CDialog(CNetFileSaveAs::IDD, pParent)
{

}

CNetFileSaveAs::~CNetFileSaveAs()
{
}

void CNetFileSaveAs::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_SAVEFILE_NAME, m_savefile_name);
}


BEGIN_MESSAGE_MAP(CNetFileSaveAs, CDialog)
END_MESSAGE_MAP()


// CNetFileSaveAs ��Ϣ�������
void CNetFileSaveAs::OnOK() 
{
	// TODO: Add extra validation here
	UpdateData(TRUE);
    m_savefile_name.TrimLeft();
	m_savefile_name.TrimRight();
	if(m_savefile_name=="")
	{
	  MessageBox("Please input the filename you want to save");
	  return;
	}
	path=path+"\\NetFiles\\"+m_savefile_name;
	path+=".net";
	CDialog::OnOK();
}