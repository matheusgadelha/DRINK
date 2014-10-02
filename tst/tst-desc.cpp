/*!
 * \brief Test program for DRINK research.
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

#include "DRINK.hpp"

using namespace cv;
using namespace std;

void selectGoodMatches( vector<DMatch>& matches, vector<DMatch>& good_matches, int dist )
{
    good_matches.clear();
    for( size_t i = 0; i<matches.size(); ++i )
    {
        if( matches[i].distance <= dist )
        {
            good_matches.push_back( matches[i] );
        }
    }
}

unsigned char testSmoothedSum(
        const cv::Mat& sum,
        const cv::KeyPoint& pt,
        int y,
        int x,
        const int _kernelSize
) {
    static const int HALF_KERNEL = _kernelSize / 2;

    int img_y = (int) (pt.pt.y) + y;
    int img_x = (int) (pt.pt.x) + x;
    int val = (sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
            - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
            - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
            + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) / ((2 * HALF_KERNEL + 1)*(2 * HALF_KERNEL + 1));
    return (unsigned char) val;
}

void showDRINKGeometry(DRINK& d, int scale, int rot) {
    Mat img = Mat::zeros(500, 500, CV_8UC3);

    for (int i = 0; i < d.ringSize * d.numRings; ++i) {
        circle(
                img,
                Point2i(img.cols / 2 + d.geometryData[i][scale][rot].x,
                img.rows / 2 + d.geometryData[i][scale][rot].y),
                1,
                Scalar(255, 255, 0)
                );
        circle(
                img,
                Point2i(img.cols / 2 + d.geometryData[i][scale][rot].x,
                img.rows / 2 + d.geometryData[i][scale][rot].y),
                d.geometryData[i][scale][rot].sigma,
                Scalar(0, 0, 255)
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

void drawDRINKGeometryAtKp(Mat& img, DRINK& d, const int scale, const int rot, const KeyPoint& pt) {
    for (int i = 0; i < d.ringSize * d.numRings; ++i) {
        circle(
                img,
                Point2i(img.cols / 2 + d.geometryData[i][scale][rot].x,
                img.rows / 2 + d.geometryData[i][scale][rot].y),
                1,
                Scalar(255, 255, 0)
                );

        circle(
                img,
                Point2i(img.cols / 2 + d.geometryData[i][scale][rot].x,
                img.rows / 2 + d.geometryData[i][scale][rot].y),
                d.geometryData[i][scale][rot].sigma,
                Scalar(0, 0, 255)
                );
    }

    for (int i = 0; i < d.numPairs; ++i) {
        float colorAtt = 255.0f;
        line(
                img,
                Point2i(img.cols / 2 + d.geometryData[d.pairs[i].a][scale][rot].x,
                img.rows / 2 + d.geometryData[d.pairs[i].a][scale][rot].y),
                Point2i(img.cols / 2 + d.geometryData[d.pairs[i].b][scale][rot].x,
                img.rows / 2 + d.geometryData[d.pairs[i].b][scale][rot].y),
                Scalar(0, colorAtt, colorAtt)
                );
    }
}

// void printDRINKProcedure( Mat& img, DRINK& d, const int scale, const int rot, const KeyPoint& pt )
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

int main(int argc, char* argv[]) {

    if (argc < 3)
        cout << "ERROR: No image path passed as argument.\n";

    const char * img_path1 = argv[1];
    const char * img_path2 = argv[2];

    Mat img_sum1, img_sum2;

    Mat img1 = imread(img_path1);
    Mat img2 = imread(img_path2);
    Mat img_matches;
    
    vector<KeyPoint> kps1, kps2;
    vector<DMatch> matches, goodMatches;;
    Mat desc1, desc2;

    Ptr<FeatureDetector> fd = new ORB();
    Ptr<DescriptorExtractor> de = new DRINK(4, false, 6, 7, 64, false);
    //Ptr<DescriptorExtractor> de = new DRINK(4, true, 6, 7, 64, false);
    //Ptr<DescriptorExtractor> de = new FREAK();
    Ptr<DescriptorMatcher> dm = new BFMatcher(NORM_HAMMING, false);
    
    fd->detect( img1, kps1 );
    de->compute( img1, kps1, desc1 );
    
    fd->detect( img2, kps2 );
    de->compute( img2, kps2, desc2 );
    
    dm->match( desc2, desc1, matches );

    for(int i=1;;++i)
    {
        selectGoodMatches( matches, goodMatches, i);
        if( goodMatches.size() >= 40 ) break;
    }
    // selectGoodMatches( matches, goodMatches, 20);

    drawMatches( img2, kps2, img1, kps1, goodMatches, img_matches, Scalar(0,200,0), Scalar(0,200,0) );
    imshow("Matches",img_matches);
    waitKey();

    return 0;

}
