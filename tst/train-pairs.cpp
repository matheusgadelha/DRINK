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

std::vector<float> columnMeanStorage;

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

inline
float stdDeviation( std::vector< std::vector<unsigned char> >& data, int col )
{
	float mean = 0.0f;
	float sqrd_sum = 0.0f;

	for( unsigned i = 0; i < data.size(); ++i )
	{
		mean += data[i][col];
	}
	mean = mean/(float)data.size();

	for( unsigned i = 0; i < data.size(); ++i )
	{
		sqrd_sum += pow(data[i][col] - mean, 2);
	}

	return sqrt(sqrd_sum/(float)data.size());
}

#define PAIRS Descriptor::pairs
#define DATA Descriptor::data
#define BEST_PAIR_NUM 64

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

float columnSum( std::vector< std::vector<unsigned char> >& data, int c )
{
	float result = 0.0f;
	for( unsigned i=0; i<data.size(); ++i )
	{
		result += data[i][c];
	}
	return result;
}

float columnMean( std::vector< std::vector<unsigned char> >& data, int c )
{
	float result = 0.0f;
	for( unsigned i=0; i<data.size(); ++i )
	{
		result += data[i][c];
	}
	return result/(float)data.size();
}

int maxStdDeviation( std::vector< std::vector<unsigned char> >& data )
{
	float current_max = 0.0f;
	int result = 0;
	for( unsigned i=0; i<data[0].size(); ++i )
	{
		float d = stdDeviation( data, i );
		if( d > current_max )
		{
			current_max = d;
			result = i;
		}
	}
	return result;
}

float correlation( std::vector< std::vector<unsigned char> >& data, int a, int b )
{
	float a_mean = columnMeanStorage[a];
	float b_mean = columnMeanStorage[b];

	float num = 0.0f;
	float den = 0.0f;

	float a_sqrd = 0.0f, b_sqrd = 0.0f;

	for( unsigned i=0; i<data.size(); ++i )
	{
		num += (data[i][a]-a_mean)*(data[i][b]-b_mean);
	}

	for( unsigned i=0; i<data.size(); ++i )
	{
		a_sqrd += pow( data[i][a]-a_mean, 2 );
		b_sqrd += pow( data[i][b]-b_mean, 2 );
	}

	den = sqrt(a_sqrd*b_sqrd);

	return num/den;
}

int main( int argc, char* argv[])
{
	Mat img;

	std::vector< std::vector<unsigned char> > data;
	std::vector< std::vector<float> > correlation_matrix ( 1722, std::vector<float>(1722,0.0f) );
	std::vector<int> bestPairs;
	std::vector<string> img_files;

	Ptr<FeatureDetector> fd = new ORB();
	Ptr<DescriptorExtractor> de = new Descriptor(4,6,7,64,true);

	vector<KeyPoint> kps;
	cv::Mat descs;

	img_files = glob("data/PNGImages/*.png");

	cout << "Processing images..." << endl;
	for( int i=0; i < img_files.size(); ++i )
	{
		img = imread( img_files[i] );
		fd->detect( img, kps );
		de->compute( img, kps, descs);
		cout << "Image " << img_files[i] << " complete." << endl;
	}

	Descriptor d = *(static_cast<Ptr<Descriptor> >(de));
	data = DATA;

	// for( int i=0; i < DATA.size(); ++ i )
	// {
	// 	for ( int j=0; j<DATA[i].size(); ++j )
	// 	{
	// 		cout << DATA[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }

	// cout << "\nTraining started. Looking for best pairs..." << endl;
	// // int firstPair = maxStdDeviation( data );
	// // bestPairs.push_back( firstPair );

	// // float t = 0.2f;
	// for( int i=0; i<DATA[0].size(); ++i)
	// 	columnMeanStorage.push_back( columnMean(DATA,i) );

	// cout << "Creating correlation matrix...";
	// for( unsigned i=0; i<correlation_matrix.size(); ++i )
	// {
	// 	for( unsigned j=i; j<correlation_matrix[i].size(); ++j )
	// 	{
	// 		// cout << "Computing correlation (" << i << "," << j << ")...";
	// 		correlation_matrix[i][j] = correlation( data, i, j );
	// 		cout << correlation_matrix[i][j] << endl;
	// 		// cout << "Done.\n";
	// 	}
	// }
	// cout << "Done\n";

	for( unsigned i=0; i<correlation_matrix.size(); ++i )
	{
		for( unsigned j=i; j<correlation_matrix[i].size(); ++j )
		{
			float in_val;
			cin >> in_val;
			correlation_matrix[j][i] = correlation_matrix[i][j] = in_val;
		}
	}
  
  int biggest_std_dev_pos = maxStdDeviation( data );
	float threshold = 0.2f;
	while( bestPairs.size() < BEST_PAIR_NUM )
	{
		int firstPair = biggest_std_dev_pos;
		bestPairs.push_back( firstPair );

		for( unsigned i=0; i<correlation_matrix.size(); ++i )
		{
			bool insert = true;
			for( unsigned j=0; j < bestPairs.size(); ++j )
			{
				if( abs(correlation_matrix[bestPairs[j]][i]) > threshold )
				{
					insert = false;
					break;
				}
			}
			if( insert )
			{
				bestPairs.push_back( i );
				cout << "Number od Best Pairs: " << bestPairs.size() << endl;
			}
			if( bestPairs.size() >= BEST_PAIR_NUM )
			{
				cout << "Enough Pairs! It's all over!!!\n";
				break;
			}
		}
		if( bestPairs.size() < BEST_PAIR_NUM )
		{
			threshold += 0.01;
			cout << "Raised threshold to " << threshold << endl;
			cout << "Best pairs: " << threshold << endl;
      
      for ( unsigned i=0; i< bestPairs.size(); ++i )
        cout << bestPairs[i] <<", ";

			bestPairs.clear();
		}
		else
		{
			break;
		}
	}

	cout << "\n\nBest Pairs: ";
	for ( unsigned i=0; i< bestPairs.size(); ++i )
		cout << bestPairs[i] <<", ";

	// cout << "Correlation Matrix:\n";
	// for( unsigned i=0; i<correlation_matrix.size(); ++i )
	// {
	// 	for( unsigned j=0; j<correlation_matrix[i].size(); ++j )
	// 	{
	// 		cout << correlation_matrix[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }

	// for( unsigned i=0; i<data[0].size(); ++i )
	// {
	// 	bool insert = true;
	// 	for( unsigned j=0; j<bestPairs.size(); ++j )
	// 	{
	// 		if( correlation( data, i, j ) > t )
	// 			insert = false;
	// 	}
	// 	cout << ".";
	// 	if( insert )
	// 	{
	// 		std::cout << "Pair " << i << " inserted." << endl;
	// 		bestPairs.push_back( i );
	// 	}
	// }

	// for( int i=0; i < DATA[0].size(); ++i )
	// {
	// 	for( int j=0; j < DATA[0].size(); ++j )
	// 	{
	// 		cout << correlation( DATA, i,j ) << " ";
	// 	}
	// 	cout << endl;
	// }

	// cout << "Number of selected pairs:" << bestPairs.size() << endl;
	// for( unsigned i=0; i<bestPairs.size(); ++i )
	// 	std::cout << bestPairs[i] << " ";
	// cout << endl;

	// Uncomment to show geometry =D
	// for( int i = 0; i < 30; ++i )
	// 	for( int j=0; j<18; ++j)
	// 		showDescriptorGeometry(*(static_cast< Ptr<Descriptor> >(de)),j,i);

	return 0;
}