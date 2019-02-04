#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {

/***********************************
Start of edited section Oct. 8, 2018
*****************************************/
//void myHist( void* bb_b, void* bb_g, void* bb_r, int w, int h )
//{
//  //Mat src;

//  /* Load image
//  src = imread( argv[1], CV_LOAD_IMAGE_COLOR);

//  if( !src.data )
//    { return -1; }*/
//printf("One\n");
//  /*Mat src;
//  src = Mat(h, w, CV_32FC3, orig);
//  src = src*255;*/
////std::cout << src << std::endl;

//  Mat srcB = Mat(h, w, CV_32FC1, bb_b);
//  srcB = srcB*255;
//  Mat srcG = Mat(h, w, CV_32FC1, bb_g);
//  srcG = srcG*255;
//  Mat srcR = Mat(h, w, CV_32FC1, bb_r);
//  srcR = srcR*255;
//printf("OneF\n");
//  /// Separate the image in 3 places ( B, G and R )
//  /*std::vector<Mat> bgr_planes;
//printf("OneS\n");
//  split( src, bgr_planes );
//printf("two\n");*/
//  /// Establish the number of bins
//  int histSize = 256;

//  /// Set the ranges ( for B,G,R) )
//  float range[] = { 0, 256 } ;
//  const float* histRange = { range };

//  bool uniform = true; bool accumulate = false;

//  Mat b_hist, g_hist, r_hist;

//printf("three\n");
//  /// Compute the histograms:
//  calcHist( &srcB, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
//  calcHist( &srcG, 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
//  calcHist( &srcR, 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
//printf("four\n");

//  // Draw the histograms for B, G and R
//  int hist_w = 512; int hist_h = 400;
//  int bin_w = cvRound( (double) hist_w/histSize );

//  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
//printf("five\n");
//  /// Normalize the result to [ 0, histImage.rows ]
//  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//printf("six\n");
//  /// Draw for each channel
//  for( int i = 1; i < histSize; i++ )
//  {
//      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
//                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
//                       Scalar( 255, 0, 0), 2, 8, 0  );
//      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
//                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
//                       Scalar( 0, 255, 0), 2, 8, 0  );
//      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
//                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
//                       Scalar( 0, 0, 255), 2, 8, 0  );
//  }
//printf("seven\n");
//  /// Display
//  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
//  imshow("calcHist Demo", histImage );
//printf("eight\n");

//  waitKey(0);
//printf("nine\n");

//}



struct Info{
  Mat hist;
  int cx;
  int cy;
  int h;
  int w;
};

struct uniqueID{
  uint16_t name;
  Info uInfo;
  //uint8_t hits;
  //time
};

#define FRAMELENGTH 200
#define ERRORTHRESH 400
//#define MAXHITS     200
#define TABLESIZE   30

struct uniqueID table[TABLESIZE];
Mat current;
Info previous;
int counter = 0;
//double test[FRAMELENGTH];
//int errH[FRAMELENGTH];
//int errW[FRAMELENGTH];
//double errC[FRAMELENGTH];
double totalError[FRAMELENGTH];
uint16_t names[FRAMELENGTH];
uint16_t allIndex[FRAMELENGTH];
uint8_t LRU[TABLESIZE];

uint16_t myHist( float* data, int iw, int ih, int w, int h, int left, int top )
{

  Rect myROI(left, top, w, h);

  Mat srcB = Mat(ih, iw, CV_32FC1, (void*)(data+2*w*h));
  srcB = srcB(myROI);
  srcB = srcB*255;
  Mat srcG = Mat(ih, iw, CV_32FC1, (void*)(data+1*w*h));
  srcG = srcG(myROI);
  srcG = srcG*255;
  Mat srcR = Mat(ih, iw, CV_32FC1, (void*)(data+0*w*h));
  srcR = srcR(myROI);
  srcR = srcR*255;


  
  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;


  /// Compute the histograms:
  calcHist( &srcB, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &srcG, 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &srcR, 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


  // Draw the histograms for B, G and R
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
//  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(b_hist, b_hist, 0, 256, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, 256, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, 256, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }

  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
  srcR = srcR/255;
  srcG = srcG/255;
  srcB = srcB/255;

  current.release();
  current.push_back(b_hist);
  current.push_back(g_hist);
  current.push_back(r_hist);

  /// Comparison of histograms

//test2

int cx;
int cy;
int errHist;
int errH;
int errW;
double errC;
double tError;

  double minError = ERRORTHRESH+ERRORTHRESH;
  int minLRU = TABLESIZE+1;
  int replaceIndex = 0;
  int minLRUIndex = TABLESIZE+1;
    for(int i = 0; i < TABLESIZE; ++i) {
      if(table[i].uInfo.hist.empty()) {
        LRU[i] = TABLESIZE;
        minLRUIndex = i;
        break;
      }
        errHist =  compareHist(table[i].uInfo.hist, current, HISTCMP_CORREL );
        errH = abs(h-table[i].uInfo.h);
        errW = abs(w-table[i].uInfo.w);
        cx = (left + (w/2));
        cy = (top - (h/2));
        errC = abs(cx-table[i].uInfo.cx) + abs(cy-table[i].uInfo.cy);
        tError = ((1-errHist)*256) + (errH) + (errW) + (errC);
      if(tError < minError) {
          minError = tError;
          replaceIndex = i;
      	}
      if(LRU[i] <= minLRU) {
          minLRU = LRU[i];
          minLRUIndex = i;
        }
      if(LRU[i] > 0) {
        LRU[i] -= 1;
      }
    }

  if(minError >= ERRORTHRESH) {
    replaceIndex = minLRUIndex;
    table[replaceIndex].name = counter;
    counter++;
  }


  table[replaceIndex].uInfo.hist = current;
  table[replaceIndex].uInfo.w = w;
  table[replaceIndex].uInfo.h = h;
  table[replaceIndex].uInfo.cx = cx;
  table[replaceIndex].uInfo.cy = cy;
  LRU[replaceIndex] = TABLESIZE;
/*  if(table[replaceIndex].hits < MAXHITS) {
    table[replaceIndex].hits++;
  }*/
  /*names[counter] = table[replaceIndex].name;
  totalError[counter] = minError;
  allIndex[counter] = replaceIndex;*/


 /*if(counter >= (FRAMELENGTH -1)) {
    counter = 0;
    for(int j = 0; j <=(FRAMELENGTH -1); ++j) {
      //std::cout << "Corr: " << test[j] << " h: " << errH[j] << " w: " << errW[j] << " c: " << errC[j] << " T: " << totalError << std::endl;
      std::cout << "T: " << totalError[j] << " N: " << names[j] << " aI: " << allIndex[j] << std::endl;
    }
    waitKey(0);
  }
  counter++;*/
return table[replaceIndex].name;

//test 1
/*
int cx;
int cy;

  if( !previous.hist.empty() ) {
    test[counter] =  compareHist(previous.hist, current, HISTCMP_CORREL );
    errH[counter] = abs(h-previous.h);
    errW[counter] = abs(w-previous.w);
    cx = (left + (w/2));
    cy = (top - (h/2));
    errC[counter] = abs(cx-previous.cx) + abs(cy-previous.cy);
  }

  previous.hist = current;
  previous.w = w;
  previous.h = h;
  previous.cx = cx;
  previous.cy = cy;

  if(counter >= (FRAMELENGTH -1)) {
    counter = 0;
    for(int j = 0; j <=(FRAMELENGTH -1); ++j) {
      //double totalError = (1/(10*test[j])) + (errH[j]) + (errW[j]) + (0.5*errC[j]);
      double totalError = ((1-test[j])*256) + (errH[j]) + (errW[j]) + (errC[j]);
      std::cout << "Corr: " << test[j] << " h: " << errH[j] << " w: " << errW[j] << " c: " << errC[j] << " T: " << totalError << std::endl;
    }
    waitKey(0);
  }
  counter++;
*/
}
/***********************************
End of edited section Oct. 8, 2018
*****************************************/

/*

IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}

Mat image_to_mat(image im)
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL); 
    if (fullscreen) {
        setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}
*/
}

#endif
