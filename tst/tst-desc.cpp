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

#include "opencv2/opencv.hpp"

#include "Descriptor.hpp"

using namespace cv;
using namespace std;

unsigned char testSmoothedSum(
	const cv::Mat& sum,
	const cv::KeyPoint& pt, 
	int y, 
	int x, 
	const int _kernelSize
){
    static const int HALF_KERNEL = _kernelSize/2;

    int img_y = (int)(pt.pt.y) + y;
    int img_x = (int)(pt.pt.x) + x;
    int val = ( sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) /((2*HALF_KERNEL+1)*(2*HALF_KERNEL+1));
    return (unsigned char) val;
}

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

void drawDescriptorGeometryAtKp( Mat& img, Descriptor& d, const int scale, const int rot, const KeyPoint& pt )
{
	for( int i = 0; i<d.ringSize * d.numRings; ++i )
	{
		circle(
			img,
			Point2i( img.cols/2 + d.geometryData[i][scale][rot].x,
					 img.rows/2 + d.geometryData[i][scale][rot].y),
			1,
			Scalar( 255, 255, 0 )
		);

		circle(
			img,
			Point2i( img.cols/2 + d.geometryData[i][scale][rot].x,
					 img.rows/2 + d.geometryData[i][scale][rot].y),
			d.geometryData[i][scale][rot].sigma,
			Scalar( 0, 0, 255 )
		);
	}

	for( int i=0; i < d.numPairs; ++i )
	{
		float colorAtt = 255.0f;
		line(
			img,
			Point2i( img.cols/2 + d.geometryData[d.pairs[i].a][scale][rot].x,
					 img.rows/2 + d.geometryData[d.pairs[i].a][scale][rot].y),
			Point2i( img.cols/2 + d.geometryData[d.pairs[i].b][scale][rot].x,
					 img.rows/2 + d.geometryData[d.pairs[i].b][scale][rot].y),
			Scalar( 0, colorAtt, colorAtt )
		);
	}
}

// void printDescriptorProcedure( Mat& img, Descriptor& d, const int scale, const int rot, const KeyPoint& pt )
// {
// 	std::cout 
// 		<< "Center: "
// 		<< (int)testSmoothedSum(
// 			img,
// 			pt,
// 			0,
// 			0,
// 			d.kernelSize
// 		);
// 	// std::cout << std::endl;

// 	// std::cout << "Geometry: " << std::endl;
// 	// for( int i = 0; i<d.ringSize * d.numRings; ++i )
// 	// {
// 	// 	std::cout << (int)testSmoothedSum(
// 	// 		img,
// 	// 		pt,
// 	// 		d.geometryData[i][scale][rot].x,
// 	// 		d.geometryData[i][scale][rot].y,
// 	// 		d.kernelSize
// 	// 	);
// 	// 	std::cout << std::endl;
// 	// }
// }

int main( int argc, char* argv[])
{
	if( argc < 3)
		cout << "ERROR: No image path passed as argument.\n";

	const char * img_path1 = argv[1];
	const char * img_path2 = argv[2];

	Mat img_sum1, img_sum2;

	cv::Mat img1 = imread( img_path1 );
	cv::Mat img2 = imread( img_path2 );
  	cv::Mat img2_scaled;

	integral( img1, img_sum1, CV_32S );
	integral( img2, img_sum2, CV_32S );

	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor(4,5,4,64,true);
	Ptr<DescriptorMatcher> dm = new cv::BFMatcher( cv::NORM_HAMMING, false );

	vector<KeyPoint> kps1;
	cv::Mat descs1;

	vector<KeyPoint> kps2;
	cv::Mat descs2;

	vector<DMatch> matches;

	fd->detect( img1, kps1 );
	de->compute( img1, kps1, descs1);

  for( double scale = 1.0; scale >= 0.25; scale = scale - 0.05 )
  {
    cv::resize(img2, img2_scaled, cv::Size(), scale, scale);

    fd->detect( img2_scaled, kps2 );
    de->compute( img2_scaled, kps2, descs2);

    dm->match(descs1, descs2, matches);

    Mat img_matches;
    drawMatches
    (
        img1, 
        kps1, 
        img2_scaled, 
        kps2,
        matches, 
        img_matches, 
        cv::Scalar(0,200,0,255), 
        cv::Scalar::all(-1),
        std::vector<char>(), 
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    cv::imshow( "Scale Cmp", img_matches );
    cv::waitKey();
  }

  //std::cout << "Number of occurences per result\n";
  //for( int i = 0; i < Descriptor::result_statistics.size(); ++i )
  //{
  //  cout << static_cast< Ptr<Descriptor> >(de)->results[i] << ": " << Descriptor::result_statistics[i] << endl;
  //}

    // std::cout << "Binary Values\n";
    // for( int i = 0; i < static_cast< Ptr<Descriptor> >(de)->bins.size(); ++i)
    // {
    // 	cout << i << ": " << static_cast< Ptr<Descriptor> >(de)->bins[i] << endl;
    // }

  //   for( int i = kps1.size(); --i; )
  //   {
	 //    circle(
		// 		img_matches,
		// 		Point2i( kps1[i].pt.x, kps1[i].pt.y ),
		// 		kps1[i].size,
		// 		Scalar( 0, 0, 255 )
		// 	); 

		// drawDescriptorGeometryAtKp(
		// 	img_matches,
		// 	*(static_cast< Ptr<Descriptor> >( de )),
		// 	round(log(kps1[i].size/Descriptor::BIGGEST_RADIUS)/log(Descriptor::SCALE_FACTOR)),
		// 	0,
		// 	kps1[i]
		// );
		// std::cout << kps1[i].size << std::endl;
  //   }

  //   for( int i = kps2.size(); --i; )
  //   {
  //   	KeyPoint kp = kps2[i];
  //   	kp.pt.x += img1.cols;

	 //    circle(
		// 		img_matches,
		// 		Point2i( kp.pt.x, kp.pt.y ),
		// 		kp.size,
		// 		Scalar( 0, 0, 255 )
		// 	); 

		// drawDescriptorGeometryAtKp(
		// 	img_matches,
		// 	*(static_cast< Ptr<Descriptor> >( de )),
		// 	round(log(kp.size/Descriptor::BIGGEST_RADIUS)/log(Descriptor::SCALE_FACTOR)),
		// 	0,
		// 	kp
		// );
		// std::cout << kps1[i].size << std::endl;
  //   }

    const int descriptor_type_size = ((float)static_cast< Ptr<Descriptor> >(de)->numBits/8)*
    									static_cast< Ptr<Descriptor> >(de)->numPairs;

 //    uchar* desc = descs1.ptr(4);
 //    for( int i=0; i < descriptor_type_size; ++i )
	// {
	//    	std::cout << std::bitset<8>(desc[i]) << " ";
	//   	// std::cout << (int)desc[i] << " ";
	// }
	// std::cout << std::endl;

	// printDescriptorProcedure(
	// 	img1,
	// 	*(static_cast< Ptr<Descriptor> >( de )),
	// 	round(log(kps1[4].size/Descriptor::BIGGEST_RADIUS)/log(Descriptor::SCALE_FACTOR)),
	// 	0,
	// 	kps1[4]
	// );
	// std::cout << std::endl;

	// desc = descs2.ptr(11);
 //    for( int i=0; i < descriptor_type_size; ++i )
	// {
	//    	std::cout << std::bitset<8>(desc[i]) << " ";
	// }
	// std::cout << std::endl;

	// printDescriptorProcedure(
	// 	img2,
	// 	*(static_cast< Ptr<Descriptor> >( de )),
	// 	round(log(kps2[11].size/Descriptor::BIGGEST_RADIUS)/log(Descriptor::SCALE_FACTOR)),
	// 	0,
	// 	kps2[11]
	// );
	// std::cout << std::endl;

	// desc = descs2.ptr(39);
 //    for( int i=0; i < descriptor_type_size; ++i )
	// {
	//    	std::cout << std::bitset<8>(desc[i]) << " ";
	//   	// std::cout << (int)desc[i] << " ";
	// }
	// std::cout << std::endl;

	// printDescriptorProcedure(
	// 	img2,
	// 	*(static_cast< Ptr<Descriptor> >( de )),
	// 	round(log(kps2[39].size/Descriptor::BIGGEST_RADIUS)/log(Descriptor::SCALE_FACTOR)),
	// 	0,
	// 	kps2[39]
	// );
	// std::cout << std::endl;


  //imshow( "Matches", img_matches );
	//waitKey();

	for( int i = 0; i < 30; ++i )
		for( int j=0; j<100; ++j)
			showDescriptorGeometry(*(static_cast< Ptr<Descriptor> >(de)),j,i);

	return 0;
}
