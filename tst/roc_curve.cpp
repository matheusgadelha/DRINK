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

void computeKeyPointsByHomography( vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, Mat& h )
{
	vector<Point2f> p1, p2;
	KeyPoint::convert( kp1, p1 );

	perspectiveTransform( p1, p2, h );

	for( size_t i = 0; i < kp1.size(); ++i )
	{
		kp2.push_back(
			KeyPoint(
				p2[i],
				kp1[i].size,
				kp1[i].angle,
				kp1[i].response,
				kp1[i].octave,
				kp1[i].class_id
			)
		);
	}
}

void createCorrectMatches( vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<DMatch>& matches, Mat& h, Mat& img )
{
	computeKeyPointsByHomography( kp1, kp2, h );
	assert( kp1.size() == kp2.size() );
	assert( kp1.size() != 0 && kp2.size() != 0 );

	vector<KeyPoint>::iterator kp1Begin = kp1.begin();
	vector<KeyPoint>::iterator kp2Begin = kp2.begin();

	for( size_t i = kp1.size(); i--; )
	{
		if( kp1[i].pt.x <= 0 || 
            kp1[i].pt.y <= 0 ||
            kp1[i].pt.x >= img.cols-0 ||
            kp1[i].pt.y >= img.rows-0 
          )
        {
          	kp1.erase( kp1Begin+i );
          	kp2.erase( kp2Begin+i );
        }
        else if( kp2[i].pt.x <= 0 || 
            kp2[i].pt.y <= 0 ||
            kp2[i].pt.x >= img.cols-0 ||
            kp2[i].pt.y >= img.rows-0 
          )
        {
          	kp1.erase( kp1Begin+i );
          	kp2.erase( kp2Begin+i );
        }
	}

	for( size_t i = 0; i < kp1.size(); ++i )
	{
		matches.push_back( DMatch( i, i, 0.0f ) );
	}
}

int main( int argc, char* argv[] )
{
	const char * img1_path = argv[1];
	const char * img2_path = argv[2];
	const char * homography_path = argv[3];

	Mat img1 = imread( img1_path );
	Mat img2 = imread( img2_path );
	ofstream results;

	vector<Point2f> recallPrecision;
	vector<KeyPoint> kp1, kp2;
	vector<DMatch> correct_matches;

	fstream homography_fstream;
	homography_fstream.open( homography_path, ios::out | ios::in );

	Mat homography( 3, 3, CV_32F );

	// Read homography from file
	for( int i=0; i<homography.rows; ++i )
	{
		for( int j=0; j<homography.cols; ++j )
		{
			homography_fstream >> homography.at<float>(i,j);
		}
	}

	//--------------DRINK

	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor(4,6,7,64,false);
	// Ptr<DescriptorExtractor> de = new ORB();
	Ptr<DescriptorMatcher> dm = new BFMatcher( cv::NORM_HAMMING, false );

	Ptr<GenericDescriptorMatcher> gdm = new VectorDescriptorMatcher( de, dm );

	fd->detect( img1, kp1 );
	// fd->detect( img2, kp2 );

	createCorrectMatches( kp1, kp2, correct_matches, homography, img1 );

	evaluateGenericDescriptorMatcher(img1, img2, homography, kp1, kp2, 0, 0, recallPrecision, gdm );

	results.open("DRINK_recallPrecision.txt");
	for( size_t i = recallPrecision.size(); i--; )
	{
		results << recallPrecision[i].x << " " << recallPrecision[i].y << endl;
	}
	cout << "END\n";
	results.close();
	results.clear();

	//--------------FREAK

	de = new FREAK(true, true);
	dm = new BFMatcher( cv::NORM_HAMMING, false );

	gdm = new VectorDescriptorMatcher( de, dm );

	fd->detect( img1, kp1 );

	evaluateGenericDescriptorMatcher(img1, img2, homography, kp1, kp2, 0, 0, recallPrecision, gdm );

	results.open("FREAK_recallPrecision.txt");
	for( size_t i = recallPrecision.size(); i--; )
	{
		results << recallPrecision[i].x << " " << recallPrecision[i].y << endl;
	}
	results.close();
	results.clear();

	//--------------ORB

	de = new ORB();
	dm = new BFMatcher( cv::NORM_HAMMING, false );

	gdm = new VectorDescriptorMatcher( de, dm );

	fd->detect( img1, kp1 );
	fd->detect( img2, kp2 );

	evaluateGenericDescriptorMatcher(img1, img2, homography, kp1, kp2, 0, 0, recallPrecision, gdm );

	results.open("ORB_recallPrecision.txt");
	for( size_t i = recallPrecision.size(); i--; )
	{
		results << recallPrecision[i].x << " " << recallPrecision[i].y << endl;
	}
	results.close();
	results.clear();

	return 0;
}