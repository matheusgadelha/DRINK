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

#include "opencv2/opencv.hpp"

using namespace std;

/*!
 * Converts int to string
 * \param n Number to be converted to string
 * \returns A string =)
 */
const string toString( int n )
{
	// Creates a string stream to use on conversion
	ostringstream a_stream;
	// Pust int into string stream
	a_stream << n;

	// Transforms string stream to string
	const string result = a_stream.str();

	return result;
}

/*!
 * Gets matrix from a file containing its values
 * \param file_path Path to a file containing matrix values
 * \returns cv::Mat structure containing file values
 */
const cv::Mat matFromFile( const string file_path )
{
	// Creates file stream
	ifstream file;
	// Opens file from path
	file.open( file_path.c_str() );

	// Creates matrix to be returned
	cv::Mat result( 3, 3, CV_32FC1 );

	// Populates matrix with file values
	file >> result.at<float>(0,0);
	file >> result.at<float>(0,1);
	file >> result.at<float>(0,2);
	file >> result.at<float>(1,0);
	file >> result.at<float>(1,1);
	file >> result.at<float>(1,2);
	file >> result.at<float>(2,0);
	file >> result.at<float>(2,1);
	file >> result.at<float>(2,2);

	return result;
}

/*!
 * Prints a matrix on the screen (test purposes only)
 * \param m A matrix to be print
 */
void printMatrix( const cv::Mat m )
{
	for( int i=0; i<m.rows; ++i )
	{
		const float* mi = m.ptr<float>(i);
		for( int j=0; j<m.cols; ++j )
		{
			cout << mi[j] << " ";
		}
		cout << endl;
	}
}

/*!
 * Creates matches for the closest points
 * \param kpts KeyPoints to be matched
 * \param pts Point2f to be matched
 */
void distanceMatching( 
	const vector<cv::KeyPoint>& kpts,
	const vector<cv::Point2f>& pts,
	vector<cv::DMatch>& matches )
{
	// Erases matches content
	matches.clear();
	// Iterates over all points in pts
	for( unsigned i=0; i < pts.size(); ++i)
	{
		// Sets inital values to be used on search at pts vector
		float smallest_dist = std::numeric_limits<float>::max();
		int closest_pt = 0;
		cv::Point2f current_pt = pts[i];

		for( unsigned j=0; j < kpts.size(); ++j)
		{
			// Tests if distance between tested points is smaller than the smallest
			// distance found so far
			if( cv::norm(kpts[j].pt - current_pt) < smallest_dist )
			{
				smallest_dist = cv::norm(kpts[j].pt - current_pt);
				closest_pt = j;
			}
		}
		// Adds new match to vector
		matches.push_back(cv::DMatch( i, closest_pt, smallest_dist ));
	}
}

/*!
 * Main test function.
 *
 * @param argc Number of args.
 * @param argv Standard content.
 */
int main( int argc, char** argv)
{
	// Tests if was not passed the correct amount of arguments
	if( argc < 2)
	{
		cerr << "ERROR >>> Missing test argument... Use one of the following:\n";
		cerr << ". . .";
		cerr << "\tbark\n";
		cerr << "\tbikes\n";
		cerr << "\tboat\n";
		cerr << "\tgraf\n";
		cerr << "\tleuven\n";
		cerr << "\ttrees\n";
		cerr << "\twall\n";

		return -1;
	}

	// Defining test folder path
	string test_folder = argv[1];
	// Adds data folder to test name to create complete test path
	// The following defines a complete test folder path to be used at processing.
	test_folder = "data/" + test_folder + "/";

	//Creates image files prefix
	string image_prefix = test_folder + "img";

	//Creates array containing all test images
	vector<cv::Mat> images;
	for( int i=1; i <= 6; ++i )
	{
		cout << image_prefix + toString(i) + ".ppm" << std::endl;
		// Creates one image from file in grayscale mode
		cv::Mat img = cv::imread( image_prefix + toString(i) + ".ppm", CV_LOAD_IMAGE_GRAYSCALE );
		// Inserts image on images array
		images.push_back( img );
	}

	// Creates feature detector and descriptors
	// OBS.: Smart pointers are used in order to easily exchange the descriptor/detector type
	cv::Ptr<cv::FeatureDetector> feature_detector = new cv::ORB();
	cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();

	// Creates keypoints/descriptors of the first image an the keypoints/descriptors
	// of the other images
	std::vector<cv::KeyPoint> first_kpts, infered_kpts;
	cv::Mat first_descs, infered_descs;

	//Detects keypoints and extracts its descriptors from the first image
	feature_detector->detect(images[0], first_kpts);
	descriptor_extractor->compute(images[0], first_kpts, first_descs);

	for( int i=1; i<6; ++i )
	{
		// Generates homography file name
		string homography_file = test_folder + "H1to" + toString(i+1) + "p";
		cout << homography_file << endl;

		//Creates homography matrix from file
		cv::Mat homography = matFromFile( homography_file );

		//Creates point2f vectors for all images
		vector<cv::Point2f> first_points, infered_points, pts_from_kpts;
		// Converts keypoints to point2f
		cv::KeyPoint::convert( first_kpts, first_points );
		// Applies homography transformation to point set
		cv::perspectiveTransform( first_points, infered_points, homography );

		// Creates vector containing point matches according to its distance
		vector<cv::DMatch> matches;
		// Detects key points on the new image
		feature_detector->detect( images[i], infered_kpts );
		// Calculates matches based on infered points to keypoints distance.
		// Since infered_points and first_points have the same size and index
		// correspondence, the resulting matches can be used to associate image1
		// points to the other image points. 
		distanceMatching( infered_kpts, infered_points, matches );

		//Draw matches
		cv::Mat matchResult;
		cv::drawMatches(images[0], first_kpts, images[i], infered_kpts, matches, matchResult);

		// Show matches
		cv::imshow( "Matches", matchResult );
		cv::waitKey(0);
	}

	// cv::imshow( "Test Image 0", images[0] );
	// cv::imshow( "Test Image 1", images[1] );
	// cv::imshow( "Test Image 2", images[2] );
	// cv::imshow( "Test Image 3", images[3] );
	// cv::imshow( "Test Image 4", images[4] );
	// cv::imshow( "Test Image 5", images[5] );

	// cv::waitKey(0);

	return 0;
}