#include <iostream>
#include <cmath>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "omp.h"
 
using namespace std;
using namespace cv;
 
 
int main()
{
 	double fps;
 	double sum_fps;
    double avg;
	double t = 0;
	double counter=1;
	char str[5];
 	Mat src, dst;
 	int gx, gy, sum;
	CvCapture* capture = 0;
 	capture = cvCaptureFromCAM( 0 );


int cofx[3][3] ={{1,0,-1},{2,0,-2},{1,0,-1}};
 int cofy[3][3] ={{1,2,1},{0,0,0},{-1,-2,-1}};


while(1) {
	
	
	
 	IplImage* image0=cvQueryFrame(capture);
	Mat m = cv::cvarrToMat(image0);
	cvtColor(m, src, CV_BGR2GRAY);
 	dst = src.clone();
	t = (double)getTickCount();
 	if( !src.data )
 		{ return -1; 
	}
	
		
	//Width and Height of the frame captured
	int width = src.cols;
	int height = src.rows;

	//Accessing the values using pointers
	uint8_t *srcptr = src.data;//Source frame pointer
	uint8_t *dstptr = dst.data;//Destination frame pointer
	
	int step = src.step;//Incremental steps between pixels
 
        // write your sobel filter here


	//Using Gaussian blur to remove unwanted noise 
	GaussianBlur(src,src, Size(3,3),0,0, BORDER_DEFAULT);
	

	omp_set_num_threads(2);//Setting the number of threads for 2 as there are 2 cores
	
	
	//Parallelizing 2 for loops with reduction 
	#pragma omp parallel for collapse(2) reduction(+:gx,gy)
	for(int x =0; x<height-2; x++)
	{
		for(int y = 0; y< width-2; y++)
		{
			gx =0;
			gy =0;

			
			//Calculate the gradient in x and y direction
			gx = (cofx[0][0] * srcptr[(x)*step+y]) + (cofx[0][1] *srcptr[(x)*step+y+1]) + (cofx[0][2] * srcptr[(x)*step+y+2]) +
              (cofx[1][0] * srcptr[(x+1)*step+y])   + (cofx[1][1] * srcptr[(x+1)*step+y+1])   + (cofx[1][2] * srcptr[(x+1)*step+y+2]) +
              (cofx[2][0] * srcptr[(x+2)*step+y]) + (cofx[2][1] * srcptr[(x+2)*step+y+1]) + (cofx[2][2] * srcptr[(x+2)*step+y+2]);

			gy = (cofy[0][0] * srcptr[(x)*step+y]) + (cofy[0][1] *srcptr[(x)*step+y+1]) + (cofy[0][2] * srcptr[(x)*step+y+2]) +
              (cofy[1][0] * srcptr[(x+1)*step+y])   + (cofy[1][1] * srcptr[(x+1)*step+y+1])   + (cofy[1][2] * srcptr[(x+1)*step+y+2]) +
              (cofy[2][0] * srcptr[(x+2)*step+y]) + (cofy[2][1] * srcptr[(x+2)*step+y+1]) + (cofy[2][2] * srcptr[(x+2)*step+y+2]);
	
		 
			//dstptr[x+1*step +y+1] = sqrt((gx*gx) + (gy*gy));
			dst.at<unsigned char>(x+1,y+1)  = sqrt((gx*gx)+(gy*gy));
		
		}
	}

 	




 	t = ((double)getTickCount() - t) / getTickFrequency();
	fps = 1.0 / t;
        sum_fps+=fps;
        avg=sum_fps/counter;
	CvFont font;
        double hScale=1.0;
        double vScale=1.0;
        int    lineWidth=1;  
        sprintf(str, "fps: %5f", avg);
	printf("Frame Rate: %f fps\n", avg);
        counter++;
 	namedWindow("Sobel Filter");
 	imshow("Sobel Filter", dst);
 
 	char key = (char) waitKey(20);
 	if(key == 27) break;
 }
 
 	return 0;
}

