#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <vector>
#include <iostream>
#include <time.h>
#include <string.h>

void medianFilter( const cv::Mat &src, cv::Mat &dst, int size );
std::vector<unsigned char> getArea(const cv::Mat &img, int x, int y, int size);
std::vector<std::vector<unsigned char> > getMatrixArea(const cv::Mat &img, 
																								 int x, int y, int size);

int main(int argc, char **argv)
{
	if( argc < 2 )
		{
			std::cout << "Image needed... " << std::endl;
			std::cout << "Program usage: ./median <image_path>" << std::endl;
			return -1;
		}
	std::cout << "\tPress 'esc' to quit." << std::endl;
	cv::namedWindow( "Median Filter" );
	int filter = 1;
	
	cv::Mat original_image = cv::imread(argv[1],0);
	cv::Mat result_image = original_image.clone();
	
	while(1)
	{
		cv::createTrackbar( "Filter size", "Median Filter", &filter, 50, NULL);
		medianFilter( original_image, result_image, filter == 0 ? 1 : filter );
		std::cout << filter << std::endl;
		cv::imshow("Median Filter", result_image);
		if(cv::waitKey() == 27)break;
	}
	
	return 0;
}

void medianFilter( const cv::Mat &src, cv::Mat &dst, int size )
{
	int width = src.cols;
	int height = src.rows;
	std::vector<unsigned char> filter;
	
	for( int y=0; y<height; ++y )
		for( int x=0; x<width; ++x )
			{
				filter = getArea( src, x, y, size );
				std::sort( filter.begin(), filter.end() );
				dst.at<unsigned char>(y,x) = filter[filter.size()/2];
				filter.clear();
			}
}

std::vector<unsigned char> getArea(const cv::Mat &img, int x, int y, int size)
{
	int width = img.cols;
	int height = img.rows;
	
	if(size % 2 == 0) --size;
	
	std::vector<unsigned char> area;
	
	int pos_x = x - size/2;
	int pos_y = y - size/2;
	
	for( int j=0; j<size; ++j )
		for( int i=0; i<size; ++i )
			if( pos_x+i >= 0 && pos_x+i < width && pos_y+j >= 0 && pos_y+j < height )
				area.push_back(img.at<unsigned char>(pos_y+j, pos_x+i));
	
	return area;
}

std::vector<std::vector<unsigned char> > getMatrixArea(const cv::Mat &img, 
																								 int x, int y, int size)
{
	int width = img.cols;
	int height = img.rows;
	
	if(size % 2 == 0) --size;
	
	std::vector<std::vector<unsigned char> > area (size, std::vector<unsigned char>(size, 0));
	
	int pos_x = x - size/2;
	int pos_y = y - size/2;
	
	for( int j=0; j<size; ++j )
		for( int i=0; i<size; ++i )
			if( pos_x+i >= 0 && pos_x+i < width && pos_y+j >= 0 && pos_y+j < height )
				area[j][i] = img.at<unsigned char>(pos_y+j, pos_x+i);
	
	return area;
}
