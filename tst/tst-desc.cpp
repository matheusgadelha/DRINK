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

double ellipse_ellipse_overlap( 
	double PHI_1, double A1, double B1, 
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
 * Transforms KeyPoints into contours applying an perspective transformation, after.
 * \param kp Set of KeyPoints
 * \param countours Vector of contours (vector<Point2f>)
 * \param H Matrix for perspective transformation.
 */
void transformKeypoints(
	const std::vector<cv::KeyPoint>& kp,
    std::vector<std::vector<cv::Point2f> >& contours,
    const cv::Mat& H )
{
    const float scale = 256.f;
    size_t i, n = kp.size();
    contours.resize(n);
    std::vector<cv::Point> temp;

    for( i = 0; i < n; i++ )
    {
        cv::ellipse2Poly(cv::Point2f(kp[i].pt.x*scale, kp[i].pt.y*scale),
                     cv::Size2f(kp[i].size*scale, kp[i].size*scale),
                     0, 0, 360, 12, temp);
        cv::Mat(temp).convertTo(contours[i], CV_32F, 1./scale);
        cv::perspectiveTransform(contours[i], contours[i], H);
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
		bool found_correct = false;
		// Stores current match
		cv::DMatch err_match = err[i];
		// Iterates over ground truth set
		for( unsigned j=0; j<gt.size(); ++j )
		{
			if( err_match.trainIdx == gt[j].trainIdx && err_match.queryIdx == gt[j].queryIdx )
			{
				found_correct = true;
				break;
			}
		}

		if( not found_correct ) false_matches++;
	}
	return false_matches;
}

/*!
 * Computes ground-truth
 * \param tk Train KeyPoints
 * \param qk Query KeyPoints
 * \param gt Ground-truth matches
 */
void computeGroundTruth(
	const vector<cv::KeyPoint>& tk, 
	const vector<cv::KeyPoint>& qk, 
	const cv::Mat homography,
	const cv::Mat img,
	vector<cv::DMatch>& gt)
{
	const float visibilityThreshold = 0.6f;
	const float overlapThreshold = 0.9f;

	// Erases matches content
	gt.clear();
	std::vector<std::vector<cv::Point2f> > tk_cont, qk_cont;

	// Transform KeyPoints into countours
	transformKeypoints(tk, tk_cont, homography);
	transformKeypoints(qk, qk_cont, cv::Mat::eye(3, 3, CV_64F));

	std::vector<cv::Rect> tk_rect, qk_rect;
	tk_rect.resize(tk_cont.size());
	qk_rect.resize(qk_cont.size());

	// Approximates contours by rectangles
	for( unsigned i=0; i < tk_cont.size(); ++i )
		tk_rect[i] = boundingRect(tk_cont[i]);

	for( unsigned i=0; i < qk_cont.size(); ++i )
		qk_rect[i] = boundingRect(qk_cont[i]);

	// Iterates over all points in pts
	for( unsigned i=0; i < tk_rect.size(); ++i)
	{
		cv::Rect r = tk_rect[i] & cv::Rect(0, 0, img.cols, img.rows );
		if( r.area() >= visibilityThreshold*tk_rect[i].area() )
		{
//         double n_area = intersectConvexConvex(kp1t_contours[i1], kp_contours[iK], noArray(), true);
//         if( n_area == 0 )
//             continue;

//         double area1 = contourArea(kp1t_contours[i1], false);
//         double area = contourArea(kp_contours[iK], false);

//         double ratio = n_area/(area1 + area - n_area);
//         n += ratio >= overlapThreshold;

			for( unsigned j=0; j < qk_rect.size(); ++j )
			{
				double inter_area = cv::intersectConvexConvex( tk_cont[i], qk_cont[j] , cv::noArray(), true );
				if( inter_area != 0)
				{
					double t_area = cv::contourArea(tk_cont[i]);
					double q_area = cv::contourArea(qk_cont[j]);

					double ratio = inter_area/( t_area + q_area - inter_area);
					if( ratio >= overlapThreshold )
					{
						float distance = cv::norm( tk[i].pt - qk[j].pt );
						gt.push_back( cv::DMatch( j, i, distance ) );
					}
				}
			}
		}
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

	std::vector< std::vector<float> > results;

	// Generates homography file name
	string homography_file = test_folder + "H1to" + toString(4) + "p";
	cout << homography_file << endl;

	//Creates homography matrix from file
	const cv::Mat homography = matFromFile( homography_file );

	// Creates vector containing point matches according to its distance
	vector<cv::DMatch> gt_matches;
	// Detects key points on the new image
	feature_detector->detect( images[3], infered_kpts );
	// Calculates matches ground-truth according to Miszolascky article.
	computeGroundTruth( first_kpts, infered_kpts, homography, images[3], gt_matches );

	//Draw matches
	cv::Mat image_matches; // Image containing matches
	cv::drawMatches(images[3], infered_kpts, images[0], first_kpts, gt_matches, image_matches);
	cv::imshow("GT Matches", image_matches);
	cv::waitKey();

	// Vector to store knn/radius matches
	vector< vector<cv::DMatch> > descriptor_matches_complete;
	// vector to sotre final descriptor matches
	vector<cv::DMatch> descriptor_matches;
	// Computes query image descriptors
	descriptor_extractor->compute(images[3], infered_kpts, infered_descs);

	// Generate results with different thresholds for matching
	for( int matching_threshold = 2; matching_threshold < 150; ++matching_threshold )
	{
		cout << matching_threshold << endl;
		// Matches and stores on 1-dimensional vector
		bf_matcher->radiusMatch( infered_descs, first_descs, descriptor_matches_complete, matching_threshold );
		for( unsigned j=0; j<descriptor_matches_complete.size(); ++j )
		{
			for( unsigned k=0; k<descriptor_matches_complete[j].size(); ++k )
			{
				descriptor_matches.push_back(descriptor_matches_complete[j][k]);
			}
		}
		// cv::drawMatches(images[3], infered_kpts, images[0], first_kpts, descriptor_matches, image_matches);
		// cv::imshow("GT Matches", image_matches);
		// cv::waitKey();

		// Computes and stores matching info
		int num_correct_matches = computeCorrectMatches( descriptor_matches, gt_matches );
		int num_false_matches = computeFalseMatches( descriptor_matches, gt_matches );
		int num_total_matches = gt_matches.size();

		cout << "-----------------------" << endl;
		cout << "Correct matches: " << num_correct_matches << endl;
		cout << "False matches: " << num_false_matches << endl;
		cout << "Total matches: " << num_total_matches << endl;
		cout << "Recall: " << num_correct_matches/(float)num_total_matches << endl;
		cout << "1-precision: " << (float)num_false_matches/(num_correct_matches+num_false_matches) << endl;
		cout << "-----------------------" << endl;
		cout << endl << "...Press any key to continue..." << endl;

		// Stores recall and precision on result vector
		vector<float> image_result;
		// 1-Precision
		image_result.push_back(
			(num_correct_matches == 0 && num_false_matches == 0) ? 0 :
			(float)num_false_matches/(num_correct_matches + num_false_matches) );
		// Recall
		image_result.push_back( num_correct_matches/(float)num_total_matches ); 
		results.push_back( image_result );

		descriptor_matches.clear();
		descriptor_matches_complete.clear();
	}

	ofstream result_file;
	result_file.open("results/matching.dat");
	for( unsigned i=0; i<results.size(); ++i )
	{
		result_file << results[i][0] << "  " << results[i][1] << endl;
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

// #include "opencv2/opencv.hpp"

// #include <stdlib.h>
// #include <stdio.h>
// #include <sys/stat.h>

// #include <limits>
// #include <cstdio>
// #include <iostream>
// #include <fstream>

// using namespace std;
// using namespace cv;

// /*
// The algorithm:

// for each tested combination of detector+descriptor+matcher:

//     create detector, descriptor and matcher,
//     load their params if they are there, otherwise use the default ones and save them

//     for each dataset:

//         load reference image
//         detect keypoints in it, compute descriptors

//         for each transformed image:
//             load the image
//             load the transformation matrix
//             detect keypoints in it too, compute descriptors

//             find matches
//             transform keypoints from the first image using the ground-truth matrix

//             compute the number of matched keypoints, i.e. for each pair (i,j) found by a matcher compare
//             j-th keypoint from the second image with the transformed i-th keypoint. If they are close, +1.

//             so, we have:
//                N - number of keypoints in the first image that are also visible
//                (after transformation) on the second image

//                N1 - number of keypoints in the first image that have been matched.

//                n - number of the correct matches found by the matcher

//                n/N1 - precision
//                n/N - recall (?)

//             we store (N, n/N1, n/N) (where N is stored primarily for tuning the detector's thresholds,
//                                      in order to semi-equalize their keypoints counts)

// */

// typedef Vec3f TVec; // (N, n/N1, n/N) - see above

// static void saveloadDDM(
// 	const string& params_filename,
//     Ptr<FeatureDetector>& detector,
//     Ptr<DescriptorExtractor>& descriptor,
//     Ptr<DescriptorMatcher>& matcher )
// {
//     FileStorage fs(params_filename, FileStorage::READ);
//     if( fs.isOpened() )
//     {
//         detector->read(fs["detector"]);
//         descriptor->read(fs["descriptor"]);
//         matcher->read(fs["matcher"]);
//     }
//     else
//     {
//         fs.open(params_filename, FileStorage::WRITE);
//         fs << "detector" << "{";
//         detector->write(fs);
//         fs << "}" << "descriptor" << "{";
//         descriptor->write(fs);
//         fs << "}" << "matcher" << "{";
//         matcher->write(fs);
//         fs << "}";
//     }
// }

// static Mat loadMat(const string& fsname)
// {
//     FileStorage fs(fsname, FileStorage::READ);
//     Mat m;
//     fs.getFirstTopLevelNode() >> m;
//     return m;
// }


// static TVec proccessMatches( Size imgsize,
//                              const vector<DMatch>& matches,
//                              const vector<vector<Point2f> >& kp1t_contours,
//                              const vector<vector<Point2f> >& kp_contours,
//                              double overlapThreshold )
// {
//     const double visibilityThreshold = 0.6;

//     // 1. [preprocessing] find bounding rect for each element of kp1t_contours and kp_contours.
//     // 2. [cross-check] for each DMatch (iK, i1)
//     //        update best_match[i1] using DMatch::distance.
//     // 3. [compute overlapping] for each i1 (keypoint from the first image) do:
//     //        if i1-th keypoint is outside of image, skip it
//     //        increment N
//     //        if best_match[i1] is initialized, increment N1
//     //        if kp_contours[best_match[i1]] and kp1t_contours[i1] overlap by overlapThreshold*100%,
//     //        increment n. Use bounding rects to speedup this step

//     int i, size1 = (int)kp1t_contours.size(), size = (int)kp_contours.size(), msize = (int)matches.size();
//     vector<DMatch> best_match(matches);
//     vector<Rect> rects1(size1), rects(size);

//     // proprocess
//     for( i = 0; i < size1; i++ )
//         rects1[i] = boundingRect(kp1t_contours[i]);

//     for( i = 0; i < size; i++ )
//         rects[i] = boundingRect(kp_contours[i]);

//     // cross-check
//     // for( i = 0; i < msize; i++ )
//     // {
//     //     DMatch m = matches[i];
//     //     int i1 = m.trainIdx, iK = m.queryIdx;
//     //     CV_Assert( 0 <= i1 && i1 < size1 && 0 <= iK && iK < size );
//     //     if( best_match[i1].trainIdx < 0 || best_match[i1].distance > m.distance )
//     //         best_match[i1] = m;
//     // }

//     int N = 0, N1 = 0, n = 0;

//     // overlapping
//     for( i = 0; i < size1; i++ )
//     {
//         int i1 = i, iK = best_match[i].queryIdx;
//         if( iK >= 0 )
//             N1++;

//         Rect r = rects1[i] & Rect(0, 0, imgsize.width, imgsize.height);
//         if( r.area() < visibilityThreshold*rects1[i].area() )
//             continue;
//         N++;

//         if( iK < 0 || (rects1[i1] & rects[iK]).area() == 0 )
//             continue;

//         double n_area = intersectConvexConvex(kp1t_contours[i1], kp_contours[iK], noArray(), true);
//         if( n_area == 0 )
//             continue;

//         double area1 = contourArea(kp1t_contours[i1], false);
//         double area = contourArea(kp_contours[iK], false);

//         double ratio = n_area/(area1 + area - n_area);
//         n += ratio >= overlapThreshold;
//     }

//     return TVec((float)N, (float)n/std::max(N1, 1), (float)n/std::max(N, 1));
// }


// static void saveResults(const string& dir, const string& name, const string& dsname,
//                         const vector<TVec>& results, const int* xvals)
// {
//     string fname1 = format("%s%s_%s_recall_precision.csv", dir.c_str(), name.c_str(), dsname.c_str());
//     // string fname2 = format("%s%s_%s_recall.csv", dir.c_str(), name.c_str(), dsname.c_str());
//     FILE* f1 = fopen(fname1.c_str(), "wt");
//     // FILE* f2 = fopen(fname2.c_str(), "wt");

//     for( size_t i = 0; i < results.size(); i++ )
//     {
//         fprintf(f1, "%f %f\n", results[i][1]*100, results[i][2]*100);
//     }
//     fclose(f1);
// }


// int main(int argc, char** argv)
// {
//     static const char* ddms[] =
//     {
//         "ORBX_BF", "ORB", "ORB", "BruteForce-Hamming",
//         //"ORB_BF", "ORB", "ORB", "BruteForce-Hamming",
//         //"ORB3_BF", "ORB", "ORB", "BruteForce-Hamming(2)",
//         //"ORB4_BF", "ORB", "ORB", "BruteForce-Hamming(2)",
//         //"ORB_LSH", "ORB", "ORB", "LSH"
//         //"SURF_BF", "SURF", "SURF", "BruteForce",
//         0
//     };

//     static const char* datasets[] =
//     {
//         "bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall", 0
//     };

//     static const int imgXVals[] = { 2, 3, 4, 5, 6 }; // if scale, blur or light changes
//     static const int viewpointXVals[] = { 20, 30, 40, 50, 60 }; // if viewpoint changes
//     static const int jpegXVals[] = { 60, 80, 90, 95, 98 }; // if jpeg compression

//     const double overlapThreshold = 0.6;

//     vector<vector<vector<TVec> > > results; // indexed as results[ddm][dataset][testcase]

//     string dataset_dir = "/home/matheusgadelha/opencv_extra-master/testdata/cv/detectors_descriptors_evaluation/images_datasets";

//     string dir=argc > 1 ? argv[1] : ".";

//     if( dir[dir.size()-1] != '\\' && dir[dir.size()-1] != '/' )
//         dir += "/";

//     int result = system(("mkdir " + dir).c_str());
//     // CV_Assert(result == 0);

//     for( int i = 0; ddms[i*4] != 0; i++ )
//     {
//         const char* name = ddms[i*4];
//         const char* detector_name = ddms[i*4+1];
//         const char* descriptor_name = ddms[i*4+2];
//         const char* matcher_name = ddms[i*4+3];
//         string params_filename = dir + string(name) + "_params.yml";

//         cout << "Testing " << name << endl;

//         Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
//         Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
//         Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_name);

//         saveloadDDM( params_filename, detector, descriptor, matcher );

//         results.push_back(vector<vector<TVec> >());

//         for( int j = 0; datasets[j] != 0; j++ )
//         {
//             const char* dsname = datasets[j];

//             cout << "\ton " << dsname << " ";
//             cout.flush();

//             const int* xvals = strcmp(dsname, "ubc") == 0 ? jpegXVals :
//                 strcmp(dsname, "graf") == 0 || strcmp(dsname, "wall") == 0 ? viewpointXVals : imgXVals;

//             vector<KeyPoint> kp1, kp;
//             vector<DMatch> matches;
//             vector<vector<Point2f> > kp1t_contours, kp_contours;
//             Mat desc1, desc;

//             Mat img1 = imread(format("%s/%s/img1.png", dataset_dir.c_str(), dsname), 0);
//             CV_Assert( !img1.empty() );

//             detector->detect(img1, kp1);
//             descriptor->compute(img1, kp1, desc1);

//             results[i].push_back(vector<TVec>());

//             for( int k = 2; ; k++ )
//             {
//                 cout << ".";
//                 cout.flush();
//                 Mat imgK = imread(format("%s/%s/img%d.png", dataset_dir.c_str(), dsname, k), 0);
//                 if( imgK.empty() )
//                     break;

//                 detector->detect(imgK, kp);
//                 descriptor->compute(imgK, kp, desc);

//                 for( int t=1; t<255; ++t )
//                 {
//                 	vector<vector<DMatch> > all_matches;
//                 	vector<DMatch> final_matches;
//                 	matcher->radiusMatch( desc, desc1, all_matches, t );
//                 	for( int i_matches = 0; i_matches<all_matches.size(); ++i_matches )
//                 	{
//                 		for( int j_matches = 0; j_matches < all_matches[i_matches].size(); ++j_matches )
//                 		{
//                 			final_matches.push_back( all_matches[i_matches][j_matches] );
//                 		}
//                 	}

// 		            Mat H = loadMat(format("%s/%s/H1to%dp.xml", dataset_dir.c_str(), dsname, k));

// 		            transformKeypoints( kp1, kp1t_contours, H );
// 		            transformKeypoints( kp, kp_contours, Mat::eye(3, 3, CV_64F));

// 		            TVec r = proccessMatches( imgK.size(), matches, kp1t_contours, kp_contours, overlapThreshold );
// 		            results[i][j].push_back(r);
// 		        }
//             }

//             saveResults(dir, name, dsname, results[i][j], xvals);
//             cout << endl;
//         }
//     }
// }