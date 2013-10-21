/*!
 * \brief Test program for descriptor research.
 * \date March 13th, 2013.
 * \version 1.0
 * \copyright GNU Public License
 *
 * \author Matheus Gadelha, Bruno Motta
 *
 *	This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <vector>
#include <bitset>
#include <glob.h>
#include <string>
#include <cmath>

#include "opencv2/opencv.hpp"

#include "Descriptor.hpp"

using namespace cv;
using namespace std;

inline std::vector<std::string> glob(const std::string& pat){
    using namespace std;
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

#define PAIRS Descriptor::pairs
#define DATA Descriptor::data

void showDescriptorGeometry( Descriptor& d, int scale, int rot)
{
	Mat img = Mat::zeros( 500, 500, CV_8UC3 );

	for( int i = 0; i<d.ringSize * d.numRings; ++i )
	{
		circle(
			img,
			Point2i( img.cols/2 + d.geometryData[i][scale][rot].x,
					 img.rows/2 + d.geometryData[i][scale][rot].y),
			1,
			Scalar( 255, 255, 0)
		);
		circle(
			img,
			Point2i( img.cols/2 + d.geometryData[i][scale][rot].x,
					 img.rows/2 + d.geometryData[i][scale][rot].y),
			d.geometryData[i][scale][rot].sigma,
			Scalar( 0, 0, 255)
		);
	}
	// for( int i=0; i < d.numPairs; ++i )
	// {
	// 	float colorAtt = 255.0f;
	// 	line(
	// 		img,
	// 		Point2i( img.cols/2 + d.geometryData[d.pairs[i*2]][scale][rot].x,
	// 				 img.rows/2 + d.geometryData[d.pairs[i*2]][scale][rot].y),
	// 		Point2i( img.cols/2 + d.geometryData[d.pairs[i*2+1]][scale][rot].x,
	// 				 img.rows/2 + d.geometryData[d.pairs[i*2+1]][scale][rot].y),
	// 		Scalar( 0, colorAtt, colorAtt )
	// 	);
	// }

	imshow("Geometry Test", img);
	waitKey();
}

float columnSum( std::vector< std::vector<int> > data, int c )
{
	float result = 0.0f;
	for( int i=0; i<data.size(); ++i )
	{
		result += data[i][c];
	}
	return result;
}

float columnMean( std::vector< std::vector<int> > data, int c )
{
	float result = 0.0f;
	for( int i=0; i<data.size(); ++i )
	{
		result += data[i][c];
	}
	return result/(float)data.size();
}

float correlation( std::vector< std::vector<int> > data, int a, int b )
{
	float a_mean = columnMean( data, a );
	float b_mean = columnMean( data, b );

	float num = 0.0f;
	float den = 0.0f;

	float a_sqrd = 0.0f, b_sqrd = 0.0f;

	for( int i=0; i<data.size(); ++i )
	{
		num += (data[i][a]-a_mean)*(data[i][b]-b_mean);
	}

	for( int i=0; i<data.size(); ++i )
	{
		a_sqrd += pow( data[i][a]-a_mean, 2 );
		b_sqrd += pow( data[i][b]-b_mean, 2 );
	}

	den = sqrt(a_sqrd*b_sqrd);

	return num/den;
}

int main( int argc, char* argv[])
{
	if( argc < 23)
		cout << "ERROR: No image path passed as argument.\n";

	const char * img_path1 = argv[1];
	const char * img_path2 = argv[2];

	Mat img_sum1, img_sum2;

	cv::Mat img1 = imread( img_path1 );
	cv::Mat img2 = imread( img_path2 );

	integral( img1, img_sum1, CV_32S );
	integral( img2, img_sum2, CV_32S );

	std::vector< std::vector<int> > data;

	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor(4,6,7,128,true);

	vector<KeyPoint> kps;
	cv::Mat descs;

	fd->detect( img1, kps );
	de->compute( img1, kps, descs);

	Descriptor d = *(static_cast<Ptr<Descriptor> >(de));

	// for( int i=0; i < DATA.size(); ++ i )
	// {
	// 	for ( int j=0; j<DATA[i].size(); ++j )
	// 	{
	// 		cout << DATA[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }

	for( int i=0; i < DATA[0].size(); ++i )
	{
		for( int j=0; j < DATA[0].size(); ++j )
		{
			cout << correlation( DATA, i,j ) << " ";
		}
		cout << endl;
	}

	cout << "Number of pairs:" << PAIRS.size();

	// Uncomment to show geometry =D
	// for( int i = 0; i < 30; ++i )
	// 	for( int j=0; j<18; ++j)
	// 		showDescriptorGeometry(*(static_cast< Ptr<Descriptor> >(de)),j,i);

	return 0;
}
