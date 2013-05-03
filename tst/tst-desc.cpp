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

 // For ellipse overlapping area
#include "program_constants.h"

using namespace std;

// Minimum distance between the found keypoint and its transformation
const float dist_tolerance = 2.0f;

// struct EllipseInfo
// {
// 	EllipseInfo( cv::Point2f _center, cv::Point2f _xAxis, cv::Point2f _yAxis)
// 	{
// 		this->center = _center;
// 		this->xAxis = _xAxis;
// 		this->yAxis = _yAxis;
// 	}

// 	cv::Point2f center;
// 	cv::Point2f xAxis;
// 	cv::Point2f yAxis;
// };

double ellipse_ellipse_overlap( double PHI_1, double A1, double B1, 
                                double H1, double K1, double PHI_2, 
                                double A2, double B2, double H2, double K2, 
                                int *rtnCode );

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

		for( unsigned j=0; j < kpts.size(); ++j )
		{
			// Tests if distance between tested points is smaller than the smallest
			// distance found so far
			if( cv::norm(kpts[j].pt - current_pt) < smallest_dist )
			{
				smallest_dist = cv::norm(kpts[j].pt - current_pt);
				closest_pt = j;
			}
		}
		// Tests if distance is small enough to consider a match and adds it to
		// matches vector
		if( smallest_dist < dist_tolerance )
		{
			matches.push_back(cv::DMatch( i, closest_pt, smallest_dist ));
		}
	}
}

/*!
 * Draws kpts with its actual size on given image
 * \param img image where points will be drawn
 * \param kpts Key points to be drawn on image
 */
void drawSizedKeyPoints( cv::Mat& img, std::vector<cv::KeyPoint> kpts )
{
	for( unsigned i=0; i < kpts.size(); ++i )
	{
		cv::ellipse( img, kpts[i].pt, cv::Size(kpts[i].size, kpts[i].size), 0, 0, 360, cv::Scalar(0,255,0));
	}
}

/*!
 * Generates ellipse information for a given set of kpts
 * \param kpts Key points used to create ellipse information.
 * Actually, will be circles, but this structure will be rearranged
 * into ellipses, further.
 * \param ellipse Result of generation
 */
 // void generateEllipsesInfoVector(
 // 	const std::vector<cv::KeyPoint>& ktps,
 // 	std::vector<EllipseInfo>& ellipses )
 // {
 // 	ellipses.resize( kpts.size() );
 // 	for( unsigned i=0; i < kpts.size(); ++i )
 // 	{
 // 		ellipses[i].x = kpts.pt.x;
 // 		ellipses[i].y = kpts.pt.y;
 // 		ellipses[i].y = kpts.pt.y;
 // 	}
 // }

/*!
 * Computes number of correct matches in err from a ground truth gt
 * \param err Matching set being tested
 * \param gt Ground truth matching set
 * \return Number of correct matches
 */
const int computeCorrectMatches( const vector<cv::DMatch>& err,  const vector<cv::DMatch>& gt)
{
	int correct_matches = 0;
	// Iterates over testet matching set
	for( unsigned i=0; i<err.size(); ++i )
	{
		// Stores current match
		cv::DMatch err_match = err[i];
		// Iterates over ground truth set
		for( unsigned j=0; j<gt.size(); ++j )
		{
			// If there is an equal match, increment correct matches
			if(err_match.trainIdx == gt[j].trainIdx && err_match.queryIdx == gt[j].queryIdx )
			{
				correct_matches++;
			}
		}
	}
	return correct_matches;
}

/*!
 * Computes number of false matches in err from a ground truth gt
 * \param err Matching set being tested
 * \param gt Ground truth matching set
 * \return Number of false matches
 */
const int computeFalseMatches( const vector<cv::DMatch>& err,  const vector<cv::DMatch>& gt)
{
	int false_matches = 0;
	// Iterates over testet matching set
	for( unsigned i=0; i<err.size(); ++i )
	{
		// Stores current match
		cv::DMatch err_match = err[i];
		// Iterates over ground truth set
		for( unsigned j=0; j<gt.size(); ++j )
		{
			// If there is an equal match, increment false matches
			if( err_match.trainIdx == gt[j].trainIdx && err_match.queryIdx != gt[j].queryIdx )
			{
				false_matches++;
			}
		}
	}
	return false_matches;
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
		// Creates one image from file in grayscale mode
		cv::Mat img = cv::imread( image_prefix + toString(i) + ".ppm", CV_LOAD_IMAGE_GRAYSCALE );
		// Inserts image on images array
		images.push_back( img );
	}

	// Creates feature detector and descriptors
	// OBS.: Smart pointers are used in order to easily exchange the descriptor/detector type
	cv::Ptr<cv::FeatureDetector> feature_detector = new cv::ORB();
	cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = new cv::ORB();
	// Descriptor matcher using hamming distance
	cv::Ptr<cv::DescriptorMatcher> bf_matcher = new cv::BFMatcher (cv::NORM_HAMMING);

	// Generic Descriptor Matcher for evaluation
	cv::Ptr<cv::GenericDescriptorMatcher> generic_matcher = new cv::VectorDescriptorMatcher( descriptor_extractor, bf_matcher );

	// Creates keypoints/descriptors of the first image an the keypoints/descriptors
	// of the other images
	std::vector<cv::KeyPoint> first_kpts, infered_kpts;
	cv::Mat first_descs, infered_descs;

	//Detects keypoints and extracts its descriptors from the first image
	feature_detector->detect(images[0], first_kpts);
	descriptor_extractor->compute(images[0], first_kpts, first_descs);

	drawSizedKeyPoints(images[0], first_kpts);
	cv::imshow("Original Image kpts", images[0]);
	cv::waitKey();

	vector< cv::Point2f > results;

	// Generates homography file name
	string homography_file = test_folder + "H1to" + toString(4) + "p";
	cout << homography_file << endl;

	//Creates homography matrix from file
	const cv::Mat homography = matFromFile( homography_file );

	//Creates point2f vectors for all images
	vector<cv::Point2f> first_points, infered_points, pts_from_kpts;
	// Converts keypoints to point2f
	cv::KeyPoint::convert( first_kpts, first_points );
	// Applies homography transformation to point set
	cv::perspectiveTransform( first_points, infered_points, homography );

	// Creates vector containing point matches according to its distance
	vector<cv::DMatch> gt_matches;
	// Detects key points on the new image
	feature_detector->detect( images[3], infered_kpts );
	// Calculates matches based on infered points to keypoints distance.
	// Since infered_points and first_points have the same size and index
	// correspondence, the resulting matches can be used to associate image1
	// points to the other image points. 
	// distanceMatching( infered_kpts, infered_points, gt_matches );

	//Draw matches
	cv::Mat image_matches; // Image containing matches
	// cv::drawMatches(images[0], first_kpts, images[i], infered_kpts, gt_matches, image_matches);

	// Vector to store knn/radius matches
	vector< vector<cv::DMatch> > descriptor_matches_complete;
	// vector to sotre final descriptor matches
	vector<cv::DMatch> descriptor_matches;
	// Computes query image descriptors
	descriptor_extractor->compute(images[3], infered_kpts, infered_descs);

	cv::evaluateGenericDescriptorMatcher(
		images[0],
		images[3],
		homography,
		first_kpts,
		infered_kpts,
		0,
		0,
		results,
		generic_matcher);

	// cv::evaluateGenericDescriptorMatcher();

	// // Generate results with different thresholds for matching
	// for( int matching_threshold = 2; matching_threshold < 256; ++matching_threshold )
	// {
	// 	cout << matching_threshold << endl;
	// 	// Matches and stores on 1-dimensional vector
	// 	bf_matcher.radiusMatch(first_descs, infered_descs, descriptor_matches_complete, matching_threshold);
	// 	for( unsigned j=0; j<descriptor_matches_complete.size(); ++j )
	// 	{
	// 		for( unsigned k=0; k<descriptor_matches_complete[j].size(); ++k )
	// 		{
	// 			descriptor_matches.push_back(descriptor_matches_complete[j][k]);
	// 		}
	// 	}

	// 	// Computes and stores matching info
	// 	int num_correct_matches = computeCorrectMatches( descriptor_matches, gt_matches );
	// 	int num_false_matches = computeFalseMatches( descriptor_matches, gt_matches );
	// 	int num_total_matches = gt_matches.size();

	// 	// cout << "-----------------------" << endl;
	// 	// cout << "Correct matches: " << num_correct_matches << endl;
	// 	// cout << "False matches: " << num_false_matches << endl;
	// 	// cout << "Total matches: " << num_total_matches << endl;
	// 	// cout << "Recall: " << num_correct_matches/(float)num_total_matches << endl;
	// 	// cout << "1-precision: " << (float)num_false_matches/(num_correct_matches+num_false_matches) << endl;
	// 	// cout << "-----------------------" << endl;
	// 	// cout << endl << "...Press any key to continue..." << endl;

	// 	// Stores recall and precision on result vector
	// 	vector<float> image_result;
	// 	// 1-Precision
	// 	image_result.push_back(
	// 		(num_correct_matches == 0 && num_false_matches == 0) ? 0 :
	// 		(float)num_false_matches/(num_correct_matches + num_false_matches) );
	// 	// Recall
	// 	image_result.push_back( num_correct_matches/(float)num_total_matches ); 
	// 	results.push_back( image_result );

	// 	descriptor_matches.clear();
	// 	descriptor_matches_complete.clear();
	// }

	ofstream result_file;
	result_file.open("results/matching.dat");
	for( unsigned i=0; i<results.size(); ++i )
	{
		result_file << results[i].x << "\t" << results[i].y << endl;
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