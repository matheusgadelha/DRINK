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

	if( argc < 2)
		cout << "ERROR: No image path passed as argument.\n";

	const char * img_path = argv[1];

	cv::Mat img = imread( img_path );
	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor();

	vector<KeyPoint> kps;
	cv::Mat descs;

	fd->detect( img, kps );
	de->compute( img, kps, descs );

	unsigned char c = 42;
	bitset<8> a(c);

	cout << (a >>= 1) << endl;
	c = a.to_ulong();
	cout << (int)c << endl;

	// showDescriptorGeometry();

	return 0;
}