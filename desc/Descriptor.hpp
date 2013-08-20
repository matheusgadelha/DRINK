#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>

#include "opencv2/opencv.hpp"

#define RBITS 16
#define PI 3.1415f

namespace cv{

	class CV_EXPORTS Descriptor : public cv::DescriptorExtractor
	{

		public:

		    // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
		   	Descriptor( int _numBits, int _ringSize, int _numRings, int _kernelSize );

		   	static void init( int _numBits = 2, int _ringSize=16, int _numRings = 8 );

		    // virtual void read( const FileNode& );
		    // virtual void write( FileStorage& ) const;

		    virtual int descriptorSize() const;
		    virtual int descriptorType() const;

		    std::vector< std::vector< std::vector<Point2i> > > geometryData;
		    std::vector< std::bitset<RBITS> > results;
		    std::vector< std::bitset<RBITS> > bins;

		    int numBits;
		    int ringSize;
		    int numRings;
		    int kernelSize;

		    int radiusStep;
		    int firstRadius;

		    static std::vector< int > result_statistics;
		    static const int scales = 8;
		    static const int rotations = 32;

		    static const double LOG2 = 0.693147180559945;
		    static const float BIGGEST_RADIUS = 111.0f;
		    static const float SMALLEST_SCALE = 0.5;
		    static const float SCALE_STEPS = 8;

		protected:
		    virtual void computeImpl(
		    	const Mat& image,
		    	vector<KeyPoint>& keypoints,
		    	Mat& descriptors ) const;

		    typedef void(*PixelTestFn)( const Mat&, const vector<KeyPoint>&, Mat& );

		    void generateGeometry();
		    void generateResults();
		    void increaseStatistics( const std::bitset<RBITS> r ) const;

		    PixelTestFn test_fn_;

	};

}



#endif