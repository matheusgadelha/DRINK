#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>
#include <cstdlib>

#include "opencv2/opencv.hpp"

#define RBITS 16
#define PI 3.1415f

namespace cv{

	class CV_EXPORTS Descriptor : public cv::DescriptorExtractor
	{
		protected:
			struct PatternPoint
			{
				PatternPoint();
				PatternPoint( int _x, int _y );

				int x;
				int y;
				int sigma;
			};


		public:

		    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
		   	Descriptor( int _numBits, int _ringSize, int _numRings, int _pairs, bool _allPairs );

			static void init( int _numBits = 2, int _ringSize=16, int _numRings = 8 );

			// virtual void read( const FileNode& );
		    // virtual void write( FileStorage& ) const;

		    virtual int descriptorSize() const;
		    virtual int descriptorType() const;

		    std::vector< std::vector< std::vector<PatternPoint> > > geometryData;
		    std::vector< std::bitset<RBITS> > results;
		    std::vector< std::bitset<RBITS> > bins;
		    std::vector<int> pairs;
		    std::vector<int> bestPairs;

		    int numBits;
		    int ringSize;
		    int numRings;
		    int numPairs;
		    int numPoints;
		    bool allPairs;

		    int radiusStep;
		    int firstRadius;

		    static std::vector< std::vector<int> > pair_result_statistics;
		    static std::vector<float> pair_std_dev;
		    static std::vector< int > result_statistics;
		    static const int scales = 8;

		    static const int ROTATION_SAMPLES = 30;
		    static const double LOG2 = 0.693147180559945;
		    static const float BIGGEST_RADIUS = 110.0f;
		    static const float SCALE_SAMPLES = 30;
		    static const float SCALE_FACTOR = 0.95f;
		    static const float GEOMETRY_SCALE_FACTOR = 0.65f;

		    float smallestRadius;

		protected:

		    virtual void computeImpl(
		    	const Mat& image,
		    	vector<KeyPoint>& keypoints,
		    	Mat& descriptors ) const;

		    typedef void(*PixelTestFn)( const Mat&, const vector<KeyPoint>&, Mat& );

		    void generateGeometry();
		    void generateResults();
		    void generateRandomPairs();
		    void generateAllPairs();
		    void increaseStatistics( const std::bitset<RBITS> r ) const;
		    void increaseStatisticsForPair( const std::bitset<RBITS> r, int p ) const;

		    PixelTestFn test_fn_;

	};

}



#endif