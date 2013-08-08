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

void showDescriptorGeometry()
{
	Mat img = Mat::zeros( 500, 500, CV_8UC3 );

	for( int i = 0; i<Descriptor::ringSize * Descriptor::numRings; ++i )
	{
		circle(
			img,
			Point2i( img.cols/2, img.rows/2 ) + Descriptor::geometryData[i],
			0,
			Scalar( 0, 0, 25 + i*20 )
		);
	}

	imshow("Geometry Test", img);
	waitKey();
}

int main( int argc, char* argv[])
{
	Descriptor::init();

	if( argc < 23)
		cout << "ERROR: No image path passed as argument.\n";

	const char * img_path1 = argv[1];
	const char * img_path2 = argv[2];

	cv::Mat img1 = imread( img_path1 );
	cv::Mat img2 = imread( img_path2 );

	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor(15);
	Ptr<DescriptorMatcher> dm = new cv::BFMatcher( cv::NORM_HAMMING, false );

	vector<KeyPoint> kps1;
	cv::Mat descs1;

	vector<KeyPoint> kps2;
	cv::Mat descs2;

	vector<DMatch> matches;

	fd->detect( img1, kps1 );
	de->compute( img1, kps1, descs1);

	fd->detect( img2, kps2 );
	de->compute( img2, kps2, descs2);

	dm->match(descs1, descs2, matches);

	Mat img_matches;
	drawMatches
    (
       	img1, 
        kps1, 
        img2, 
        kps2,
        matches, 
        img_matches, 
        cv::Scalar(0,200,0,255), 
        cv::Scalar::all(-1),
        std::vector<char>(), 
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    std::cout << "Number of occurences per result\n";
    for( int i = 0; i < Descriptor::result_statistics.size(); ++i)
    {
    	cout << Descriptor::results[i] << ": " << Descriptor::result_statistics[i] << endl;
    }

    std::cout << "Positive Binary Values\n";
    for( int i = 0; i < Descriptor::positiveBin.size(); ++i)
    {
    	cout << i << ": " << Descriptor::positiveBin[i] << endl;
    }

    imshow("Matches", img_matches);
	waitKey();
	// showDescriptorGeometry();

	return 0;
}