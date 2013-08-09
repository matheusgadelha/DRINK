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

		    Point2i* geometryData;
		    std::vector< std::bitset<RBITS> > results;
		    std::vector< std::bitset<RBITS> > bins;

		    int numBits;
		    int ringSize;
		    int numRings;
		    int kernelSize;

		    static std::vector< int > result_statistics;
		    static const int firstRadius = 5;
		    static const int radiusStep = 5;

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