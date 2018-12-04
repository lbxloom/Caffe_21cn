#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMainWindow>
#include <QUrl>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QNetworkAccessManager>
#include <QString>
#include <QNetworkProxy>
#include <QEventLoop>
#include <QUrlQuery>
#include <QNetworkCookieJar>
#include <QBuffer>
#include <QImageReader>
#include <QDir>
#include <windows.h>

typedef int(__stdcall *PFN_21CNOCR)(BYTE* pByte, int nByteSize, char* pCode);

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void init();

    void getNext();

    void showImage(QByteArray& bytes);

    void getImage(QUrl redirectedUrl);

    void reDirect(QUrl redirectedUrl);

    bool initDll();

private slots:
    void on_pushButton_clicked();
    //void onEnter();
private:
    Ui::MainWindow *ui;
    QNetworkAccessManager m_manager;
    QByteArray m_byteRpl;
    QImage m_curImage;
    QString m_token;
    qint64 m_nTime;
    char* m_pImgBuf;
    int m_ImgLen;
    PFN_21CNOCR m_pfn;
};

#endif // MAINWINDOW_H
