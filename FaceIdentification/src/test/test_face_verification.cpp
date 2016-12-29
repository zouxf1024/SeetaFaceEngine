/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Identification module, containing codes implementing the
 * face identification method described in the following paper:
 *
 *
 *   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
 *   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
 *   In Frontiers of Computer Science.
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include<iostream>
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

//#include <opencv/cv.h>
//#include <opencv/highgui.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/types_c.h"

#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <dirent.h>

#include <pthread.h>
#include <unistd.h>

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./image/";
std::string MODEL_DIR = "./";
std::string FEA_DIR = "./fea/";
#endif

int getImgPoints(seeta::FaceDetection* pDetector,
        seeta::FaceAlignment* pPoint_detector,
        FaceIdentification* pFace_recognizer,
        cv::Mat& img_color,
        seeta::FacialLandmark* points)
{

    cv::Mat img_gray;

    try 
    {
        cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);
    }
    catch (cv::Exception exce) 
    {
        return 0;
    }

    ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
    img_data_gray.data = img_gray.data;

    // Detect faces
    std::vector<seeta::FaceInfo> faces = pDetector->Detect(img_data_gray);
    int32_t face_num = static_cast<int32_t>(faces.size());

    if (face_num == 0 )
    {
        std::cout << "Faces are not detected." << endl;
        return 0;
    }

    // Detect 5 facial landmarks
    pPoint_detector->PointDetectLandmarks(img_data_gray, faces[0], points);

    return 1;
}
int getImgFea(seeta::FaceDetection* pDetector,
        seeta::FaceAlignment* pPoint_detector,
        FaceIdentification* pFace_recognizer,
        cv::Mat& img_color,
        float* fea)

{
    seeta::FacialLandmark points[5];
    if ( !getImgPoints(pDetector, pPoint_detector, pFace_recognizer, img_color, points) )
        return 0;

    ImageData img_data_color(img_color.cols, img_color.rows, img_color.channels());
    img_data_color.data = img_color.data;

    // Extract face identity feature
    pFace_recognizer->ExtractFeatureWithCrop(img_data_color, points, fea);
	return 1;
}

int saveImgFeaToFile(seeta::FaceDetection* pDetector,
        seeta::FaceAlignment* pPoint_detector,
        FaceIdentification* pFace_recognizer,
        cv::Mat& img_color,
        const std::string filename)

{
    seeta::FacialLandmark points[5];
    if ( !getImgPoints(pDetector, pPoint_detector, pFace_recognizer, img_color, points))
        return 0;

    ImageData img_data_color(img_color.cols, img_color.rows, img_color.channels());
    img_data_color.data = img_color.data;

    // Extract face identity feature
    pFace_recognizer->ExtractFeatureWithCropToFile(img_data_color, points, filename);
	return 1;
}

void imageShowString(cv::Mat& img_color, std::string text)
{
    CvFont font;
    cvInitFont( &font, 6 , 1, 1, 0, 2 );
    //cvInitFont( &font, FONT_HERSHEY_SCRIPT_SIMPLEX, 2, 2, 0, 3 );
    //int x = (points[0].x + points[1].x) / 2;
    //int y = points[0].y - (points[2].y - points[1].y);
    int x = 100;//(points[0].x + points[1].x) / 2;
    int y = 100;//points[0].y - (points[2].y - points[1].y);
    CvPoint textPos =cvPoint(x,y);

    CvMat img = img_color;
    cvPutText( &img, text.c_str(), textPos, &font, cvScalar(0,255,255));

    cv::imshow("zouxf",img_color);
}
void imageShow(cv::Mat& img_color, float sim)
{
    char simtext[15];
    sprintf(simtext,"%f",sim);
    imageShowString(img_color, std::string(simtext));
}

size_t getBlodsFromDir(std::string fea_dir, std::vector<Blob>& blobs, std::vector<std::string> &names)
{
    struct dirent *dir;
    DIR* pDir = opendir(fea_dir.c_str());
    if (pDir)
    {
        FILE* file = nullptr;
        std::string filename;
        while ((dir = readdir(pDir)) != 0) 
        {
            //if (dir->d_type == DT_CHR)
            {
                filename = std::string(dir->d_name);
                if (filename.size() < 4)
                    continue;
                if (!filename.compare(filename.size()-3,3,"fea"))
                {
                    cout << "get " << filename << " fea" << endl;
                    std::string fullPath = fea_dir + filename;
                    fopen_s(&file, fullPath.c_str(), "rb");
                    if (file == nullptr) 
                    {
                      LOG(ERROR) << filename << " not exist!";
                      continue;
                    }
                    
                    blobs.push_back(Blob(file));
                    names.push_back(filename.substr(0,filename.size()-4));
                    fclose(file);
                }
            }
        }
        closedir(pDir);
    }
    return blobs.size();
}

#define MAX_THREAD    10
#define DEFAULT_CAM_INDEX 0
#define DEFAULT_VERI_THRESHOLD 0.6

//#define SHOWIMG
//#define ENABLE_CACLSIM_THREAD

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_rwlock_t feaLock;

pthread_cond_t feacond;
pthread_mutex_t feamutex;

cv::Mat frame;
int getit=0;
std::string getName;

std::vector<Blob> blobs;
std::vector<std::string> names;
float probe_fea[2048];
int getVaildFea = 0;

FaceIdentification* pface_recognizer;
float gVeriThreshold = DEFAULT_VERI_THRESHOLD;

struct CACLPARA
{
    FaceIdentification * pface_recognizer;
    int galleryBlobsIndex;
    float *fea;
    int & getVaildFea;
};

typedef struct CACLPARA CaclPara;

#ifdef ENABLE_CACLSIM_THREAD
void * caclSimThread(void * arg)
{
    float sim;
    CaclPara & mCaclPara = *((CaclPara *)arg);
    int index = mCaclPara.galleryBlobsIndex;
    float *probe_fea = mCaclPara.fea;
    int gallerySize = blobs.size();

    FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());

    while (1)
    {
        //std::cout << " caclSimThread " << index << " wait cond" << endl;
        pthread_cond_wait(&feacond, &feamutex);

        if (mCaclPara.getVaildFea > 0) 
        {
            //std::cout << " caclSimThread " << index << " get cond " << endl;
            pthread_rwlock_rdlock(&feaLock);
            //std::cout << " caclSimThread " << index << " get rd lock" << endl;
            //sim = mCaclPara.pface_recognizer->CalcSimilarity(blobs.at(index).data().get(), probe_fea);
            for (int i = index; i < gallerySize; i += MAX_THREAD) 
            {
                sim = face_recognizer.CalcSimilarity(blobs.at(i).data().get(), probe_fea);
                //std::cout << names.at(i) << " is " << sim << " at " << i << " in thread " << index << endl;

                if (sim > gVeriThreshold) {
                    getit = 1;
                    getName = names.at(i);
                    std::cout << getName << " is verificate, sim = " << sim << endl;
                }
            }
            //std::cout << " caclSimThread " << index << " d unlock" << endl;
            pthread_rwlock_unlock(&feaLock);
        }
    }
}
#endif

void * getFeaThread(void * arg)
{
    float sim;

    // Initialize face detection model
    seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
    detector.SetMinFaceSize(40);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);

    // Initialize face alignment model 
    seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");

    // Initialize face Identification model 
    pface_recognizer = new FaceIdentification((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
    FaceIdentification& face_recognizer = *pface_recognizer;
    //FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());

    std::string fea_dir = FEA_DIR;

    size_t gallerySize = getBlodsFromDir(fea_dir, blobs, names);
    cout << "get " << gallerySize << " fea" << endl;

    if (gallerySize > 0)
    {
        //pthread_t caclSimTid[gallerySize];
        pthread_t caclSimTid[MAX_THREAD];

        pthread_rwlock_init(&feaLock, NULL);

        //for (int i=0; i<gallerySize; i++)
#ifdef ENABLE_CACLSIM_THREAD
        for (int i=0; i<MAX_THREAD && i<gallerySize; i++)
        {
            CaclPara para = {pface_recognizer, i, probe_fea, getVaildFea};
            pthread_create(&caclSimTid[i], NULL, caclSimThread, (void *)(&para));
        }
#endif

        int i =0;
        const int frameWidth = 640;
        const int frameHeight = 480;
        const double scale = 0.5;
        const cv::Size dsize = cv::Size(frameWidth * scale, frameHeight * scale);
        cv::Mat probe_img_color = cv::Mat(dsize, CV_32S);

        while (1)
        {
            //std::cout << " getFeaThread request lock" << endl;
            pthread_mutex_lock(&mutex);
            //std::cout << " getFeaThread get lock" << endl;
            //std::cout << " getFeaThread wait signal" << endl;
            pthread_cond_wait(&cond,&mutex);
            //std::cout << " getFeaThread get signal" << endl;
            try
            {
                cv::resize(frame, probe_img_color, dsize);
            }
            catch (cv::Exception e)
            {
                pthread_mutex_unlock(&mutex);
                std::cout << " resize fail" << endl;
                continue;
            }
            pthread_mutex_unlock(&mutex);
            //std::cout << " getFeaThread Unlock" << endl;

            //std::cout << " getFeaThread wr lock" << endl;
            pthread_rwlock_wrlock(&feaLock);
            getit = 0;
            getVaildFea = getImgFea(&detector,&point_detector,&face_recognizer, probe_img_color,probe_fea);
            //std::cout << " getFeaThread wr unlock " << getVaildFea << endl;
            pthread_rwlock_unlock(&feaLock);
            if (getVaildFea > 0)
            {
                //std::cout << " getFeaThread broadcast cond" << endl;
                pthread_cond_broadcast(&feacond);
            }
            else
                std::cout << " no vaild face cond" << endl;
#ifndef ENABLE_CACLSIM_THREAD
            if (getVaildFea > 0)
            {
                for (i=0; i<gallerySize; i++)
                {
                    sim = face_recognizer.CalcSimilarity(blobs.at(i).data().get(), probe_fea);
                    //std::cout << names.at(i) << " is " << sim << endl;
                    if (sim > gVeriThreshold)
                        break;
                }
            }
            if (sim > gVeriThreshold) {
                getit = 1;
                getName = names.at(i);
                std::cout << getName << " is verificate" << endl;
            }
            else
            {
                std::cout << " nobody is verificate" << endl;
                getit = 0;
            }
#endif
        }
    }
}

void * imgShowThread(void * arg)
{
    cv::VideoCapture * capture = (cv::VideoCapture *)arg;

    while (1) 
    {
        //std::cout << " ======== imgShowThread request lock ==============" << endl;
        pthread_mutex_lock(&mutex);
        //std::cout << " imgShowThread get lock" << endl;
        (*capture) >> frame;
        //std::cout << " imgShowThread send signal" << endl;
        pthread_cond_signal(&cond);
        //std::cout << " imgShowThread send signal finish" << endl;
        pthread_mutex_unlock(&mutex);
        //std::cout << " imgShowThread Unlock" << endl;

#ifdef SHOWIMG
        imageShowString(frame,getName);

        int c = cv::waitKey(5);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
#else
        usleep(10000);
#endif
    }
}

int main(int argc, char* argv[]) 
{
    cv::VideoCapture capture;
    cv::Mat frame;
    cv::Mat gallery_img_color;
    float sim;
    float gallery_fea[2048];
    float probe_fea[2048];

    if (!strcmp("fea",argv[1]))
    {
        // Initialize face detection model
        seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
        detector.SetMinFaceSize(40);
        detector.SetScoreThresh(2.f);
        detector.SetImagePyramidScaleFactor(0.8f);
        detector.SetWindowStep(4, 4);

        // Initialize face alignment model 
        seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");

        // Initialize face Identification model 
        FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
        std::string image_dir = DATA_DIR;
        std::string fea_dir = FEA_DIR;


        struct dirent *dir;
        DIR* pDir = opendir(image_dir.c_str());
        if (pDir) 
        {
            while ((dir = readdir(pDir)) != 0) 
            {
                //if (dir->d_type == DT_CHR)
                {
                    std::string filename = std::string(dir->d_name);
                    if (filename.size() < 4)
                        continue;
                    if (!(filename.compare(filename.size()-3,3,"jpg") & filename.compare(filename.size()-3,3,"JPG")))
                    {
                        cout << "save " << filename << " fea" << endl;
                        gallery_img_color = cv::imread(image_dir + filename,1);
                        const std::string dataFileName = fea_dir + filename.replace(filename.length()-3,3,"fea");
                        saveImgFeaToFile(&detector, &point_detector, &face_recognizer, gallery_img_color, dataFileName);
                    }
                }
            }
            closedir(pDir);
        }
    }
    else if (!strcmp("cam",argv[1]))
    {
        int camIndex = DEFAULT_CAM_INDEX;
        if ( argc > 2) 
        {
            camIndex = argv[2][0] - '0';
            if (argc > 3)
            {
                char *pEndc;
                float threshold = strtof(argv[3], &pEndc);

                if ((threshold > 0) && (threshold < 1)) 
                {
                    gVeriThreshold = threshold;
                }
            }
        }

        std::cout << "Open camera " << camIndex << ", Set threshold " << gVeriThreshold << endl;

        if ( !capture.open(camIndex))
        {
            cout << "Capture from camera didn't work" << endl;
            return -1;
        }

        if( capture.isOpened() )
        {
            pthread_t  imgTid, feaTid;
            void * imgThreadRe;
            void * feaThreadRe;


            pthread_cond_init(&cond, NULL);
            pthread_mutex_init(&mutex, NULL);

            pthread_cond_init(&feacond, NULL);
            pthread_mutex_init(&feamutex, NULL);

            pthread_create(&imgTid, NULL, imgShowThread, (void *)(&capture));
            //pthread_create(&imgTid, NULL, imgShowThread, NULL);
            pthread_create(&feaTid, NULL, getFeaThread, NULL);

            pthread_join(imgTid, &imgThreadRe);
        }
    }
    else if (!strcmp("camera",argv[1]))
    {
        // Initialize face detection model
        seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
        detector.SetMinFaceSize(40);
        detector.SetScoreThresh(2.f);
        detector.SetImagePyramidScaleFactor(0.8f);
        detector.SetWindowStep(4, 4);

        // Initialize face alignment model 
        seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");

        // Initialize face Identification model 
        FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
        std::string image_dir = DATA_DIR;
        std::string fea_dir = FEA_DIR;

        gallery_img_color = cv::imread(image_dir + "g_0001.jpg", 1);
        if ( !getImgFea(&detector,&point_detector,&face_recognizer,gallery_img_color,gallery_fea))
        {
            cout << "no face detected in gallery image" << endl;
            return -1;
        }

        if ( !capture.open(0))
        {
            cout << "Capture from camera didn't work" << endl;
            return -1;
        }

        if( capture.isOpened() )
        {
            const int frameWidth = 640;
            const int frameHeight = 480;
            const double scale = 0.5;

            const cv::Size dsize = cv::Size(frameWidth * scale, frameHeight * scale);
            cv::Mat probe_img_color = cv::Mat(dsize, CV_32S);

            while (1) 
            {
                sim = 0;
                capture >> frame;
                cv::resize(frame, probe_img_color, dsize);

                if (getImgFea(&detector,&point_detector,&face_recognizer, probe_img_color,probe_fea))
                {
                    sim = face_recognizer.CalcSimilarity(gallery_fea, probe_fea);
                }
                //frame = probe_img_color.clone();
                imageShow(probe_img_color,sim);

                int c = cv::waitKey(10);
                if( c == 27 || c == 'q' || c == 'Q' )
                    break;
            }
        }
    }

    std::cout << sim <<endl;
    cv::waitKey();

    return 0;
}


