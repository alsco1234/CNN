#include <iostream>
#include <opencv\cv.h>
#include <opencv\highgui.h>

using namespace std;
using namespace cv;

int main()
{
    const string inPath = " "; //파일명 포함한 입력파일 경로
    string outPath; 
    string folderPath = " ";//저장할 폴더경로
    int fileCnt = 1;
    int frameCnt = 0;

    VideoCapture vc(inPath);
    if (!vc.isOpened()){
        cout << "can't open File" << endl;
        return 0;
    }
    double fps = vc.get(CV_CAP_PROP_FPS);
    int width = (int)vc.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int)vc.get(CV_CAP_PROP_FRAME_HEIGHT);
    outPath = folderPath + "/*파일이름*/" +to_string(fileCnt) + ".avi";
    cout << "outputPath" << outPath << endl;
    VideoWriter vw;
    vw = VideoWriter(outPath, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height));
    while (1)
    {
        Mat frame;
        vc >> frame;
        if (frame.empty()){
            vw.release();
            cout << "done" << endl;
            return 0;
        }
        if (frameCnt == 1800){ //fps 30기준으로 프레임갯수가 1800개이면 1분!
            vw.release();
            fileCnt++;
            frameCnt = 0;
            outPath = folderPath + "/*파일이름*/" + to_string(fileCnt) + ".avi";
            cout << "outputPath" << outPath << endl;
            vw = VideoWriter(outPath, CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height));
        }
        vw << frame;
        frameCnt++;
        
    }
}