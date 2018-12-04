#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QMessageBox>
#include <windows.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFixedSize(width(), height());
    //setWindowFlags(windowFlags()& Qt::WindowMaximizeButtonHint & ~Qt::WindowMinimizeButtonHint);

    m_pfn = nullptr;
    m_pImgBuf = nullptr;

    ui->lineEdit_2->setValidator(new QIntValidator(0, 10000, this));

    m_manager.setCookieJar(new QNetworkCookieJar(this));

    //设置代理 拦截包
//    QNetworkProxy proxy;
//    proxy.setType(QNetworkProxy::HttpProxy);
//    proxy.setHostName("127.0.0.1");
//    proxy.setPort(8080);
//    m_manager.setProxy(proxy);

    QDir dir;
    if(!dir.exists("PngSaved"))
    {
        bool bRet = dir.mkpath("PngSaved");
        if(!bRet)
        {
            return;
        }
    }
    if(!initDll())
    {
       QMessageBox box;
       box.setText("找不到库");
       box.exec();
       return;
    }
    init();
    getNext();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showImage(QByteArray& bytes)
{
    m_ImgLen = bytes.count();
    m_pImgBuf = new char[m_ImgLen];
    memcpy(m_pImgBuf, bytes.data(), m_ImgLen);
    QBuffer buffer(&bytes);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer,"PNG");
    m_curImage = reader.read();
    if(!m_curImage.isNull())
    {
        QPixmap pix = QPixmap::fromImage(m_curImage);
        ui->label->setPixmap(pix);
    }
}

void MainWindow::getImage(QUrl redirectedUrl)
{
    //获取验证码
    //创建url
    QString baseUrl = QString("https://open.e.189.cn/api/logbox/oauth2/picCaptcha.do?token=%1&rnd=%2").arg(m_token).arg(m_nTime);
    QUrl url(baseUrl);


    // 构造请求
    QNetworkRequest request;
    request.setUrl(url);
    request.setRawHeader("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0");
    request.setRawHeader("Accept","text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    request.setRawHeader("Accept-Language","zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2");
    request.setRawHeader("Accept-Encoding","gzip, deflate");
    request.setRawHeader("Referer", redirectedUrl.toString().toUtf8());
    request.setRawHeader("Connection","close");
    request.setRawHeader("Upgrade-Insecure-Requests","1");


    //认证SSL 支持HTTPS
    QSslConfiguration config;
    config.setPeerVerifyMode(QSslSocket::VerifyNone);
    config.setProtocol(QSsl::TlsV1_1);
    request.setSslConfiguration(config);

    //接收结果
    QNetworkReply *pReplay = m_manager.get(request);

    //同步
    QEventLoop eventLoop;
    QObject::connect(&m_manager, &QNetworkAccessManager::finished, &eventLoop, &QEventLoop::quit);
    eventLoop.exec();

    // 获取响应信息
    m_byteRpl = pReplay->readAll();
    showImage(m_byteRpl);
    pReplay->deleteLater();
    pReplay = Q_NULLPTR;
}

void MainWindow::reDirect(QUrl redirectedUrl)
{
    // 构造请求
    QNetworkRequest request;
    request.setUrl(redirectedUrl);
    request.setRawHeader("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0");
    request.setRawHeader("Accept","text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    request.setRawHeader("Accept-Language","zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2");
    request.setRawHeader("Accept-Encoding","gzip, deflate");
    request.setRawHeader("Referer","http://mail.21cn.com/w2/");
    request.setRawHeader("Connection","close");
    request.setRawHeader("Upgrade-Insecure-Requests","1");

    //认证SSL 支持HTTPS
    QSslConfiguration config;
    config.setPeerVerifyMode(QSslSocket::VerifyNone);
    config.setProtocol(QSsl::TlsV1_1);
    request.setSslConfiguration(config);

    //接收结果
    QNetworkReply *pReplay = m_manager.get(request);

    //同步
    QEventLoop eventLoop;
    QObject::connect(&m_manager, &QNetworkAccessManager::finished, &eventLoop, &QEventLoop::quit);
    eventLoop.exec();

    // 获取响应信息
    m_byteRpl = pReplay->readAll();

    //设置token
    QString str(m_byteRpl);
    QString subStr = "token=";
    int nIndex = str.indexOf(subStr);
    m_token = str.mid(nIndex, 41);
    qDebug() << m_token;

    pReplay->deleteLater();
    pReplay = Q_NULLPTR;
}

bool MainWindow::initDll()
{
    HMODULE h21CNDll = LoadLibraryA("21cnOCR.dll");
    if (h21CNDll == NULL)
    {
        return false;
    }

    m_pfn = (PFN_21CNOCR)GetProcAddress(h21CNDll, "OCR_E");
    if (m_pfn == NULL)
    {
        return false;
    }
    return true;
}

//void MainWindow::onEnter()
//{
//    if(ui->lineEdit->text() == "")
//    {
//        return;
//    }

//    m_curImage.save(QString("PngSaved\\%1.png").arg(ui->lineEdit->text()));
//    ui->lineEdit->clear();

//    getNext();
//}

void MainWindow::init()
{
    //创建url
    QString baseUrl = "https://mail.21cn.com/w2/";
    QUrl url(baseUrl);

    // 构造请求
    QNetworkRequest request;
    request.setUrl(url);
    request.setRawHeader("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0");
    request.setRawHeader("Accept","text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    request.setRawHeader("Accept-Language","zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2");
    request.setRawHeader("Accept-Encoding","gzip, deflate");
    request.setRawHeader("Referer","http://mail.21cn.com/");
    request.setRawHeader("Connection","close");
    request.setRawHeader("Upgrade-Insecure-Requests","1");

    //认证SSL 支持HTTPS
    QSslConfiguration config;
    config.setPeerVerifyMode(QSslSocket::VerifyNone);
    config.setProtocol(QSsl::TlsV1_1);
    request.setSslConfiguration(config);

    //接收结果
    QNetworkReply *pReplay = m_manager.get(request);

    //同步
    QEventLoop eventLoop;
    QObject::connect(&m_manager, &QNetworkAccessManager::finished, &eventLoop, &QEventLoop::quit);
    eventLoop.exec();

    // 获取响应信息
    m_byteRpl = pReplay->readAll();

    pReplay->deleteLater();
    pReplay = Q_NULLPTR;
}

void MainWindow::getNext()
{
    //创建url
    if(m_pImgBuf != nullptr)
    {
        delete m_pImgBuf;
        m_pImgBuf = nullptr;
    }

    QDateTime time = QDateTime::currentDateTime();   //获取当前时间
    m_nTime = time.toMSecsSinceEpoch();

    QString baseUrl = QString("https://mail.21cn.com/w2/logon/UnifyLogin.do?t=%1").arg(m_nTime);
    QUrl url(baseUrl);


    // 构造请求
    QNetworkRequest request;
    request.setUrl(url);
    request.setRawHeader("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0");
    request.setRawHeader("Accept","text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    request.setRawHeader("Accept-Language","zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2");
    request.setRawHeader("Accept-Encoding","gzip, deflate");
    request.setRawHeader("Referer","http://mail.21cn.com/w2/");
    request.setRawHeader("Connection","close");
    request.setRawHeader("Upgrade-Insecure-Requests","1");


    //认证SSL 支持HTTPS
    QSslConfiguration config;
    config.setPeerVerifyMode(QSslSocket::VerifyNone);
    config.setProtocol(QSsl::TlsV1_1);
    request.setSslConfiguration(config);

    //接收结果
    QNetworkReply *pReplay = m_manager.get(request);

    //同步
    QEventLoop eventLoop;
    QObject::connect(&m_manager, &QNetworkAccessManager::finished, &eventLoop, &QEventLoop::quit);
    eventLoop.exec();

    // 获取响应信息
    QVariant redirectionTarget = pReplay->attribute(QNetworkRequest::RedirectionTargetAttribute);
    pReplay->deleteLater();
    pReplay = Q_NULLPTR;
    if (!redirectionTarget.isNull())
    {
        QUrl redirectedUrl = url.resolved(redirectionTarget.toUrl());

        //访问重定向网址
        reDirect(redirectedUrl);

        //获取验证码
        getImage(redirectedUrl);
    }
}

void MainWindow::on_pushButton_clicked()
{
    ui->pushButton->setEnabled(false);
    ui->lineEdit_2->setReadOnly(true);
    int n = ui->lineEdit_2->text().toInt();
    int i = 0;
    while(i < n)
    {
        //顺序命名
        //m_curImage.save(QString("PngSaved\\%1.png").arg(i+1));

        ui->lineEdit_2->setText(QString("%1").arg(i+1));

        //自动识别接口
        char aryCode[256] = {0};
        m_pfn((BYTE*)m_pImgBuf, m_ImgLen, aryCode);

        m_curImage.save(QString("PngSaved/%1.png").arg(aryCode));

        //获取下一张
        getNext();

        i++;
    }
    ui->lineEdit_2->setReadOnly(false);
    ui->pushButton->setEnabled(true);
}
