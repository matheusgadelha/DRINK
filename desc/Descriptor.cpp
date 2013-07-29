#include "Descriptor.hpp"

inline 
unsigned char valueAt( const cv::KeyPoint& kp, cv::Point2i p , cv::Mat& img )
{
	return img.at<unsigned char>( p.y + kp.pt.y, p.x + kp.pt.x );
}

inline
unsigned char valueAtCenter( const cv::KeyPoint& kp, cv::Mat& img )
{
	return img.at<unsigned char>( kp.pt.y, kp.pt.x );
}

namespace cv{

	int Descriptor::numBits;
	int Descriptor::ringSize;
	int Descriptor::numRings;

	const int Descriptor::firstRadius;
	const int Descriptor::radiusStep;
	const int Descriptor::kernelSize;

	Point2i* Descriptor::geometryData;
	std::vector< std::bitset<16> > Descriptor::results;

	Descriptor::Descriptor()
	{ }

	void Descriptor::init( int _numBits, int _ringSize, int _numRings )
	{
		numBits  = _numBits;
		ringSize = _ringSize;
		numRings = _numRings;

		geometryData = new Point2i[ringSize*numRings];
		generateGeometry();
		generateResults();
	}

	void Descriptor::generateGeometry()
	{
		for( int i_ring = 0; i_ring < numRings; ++i_ring )
		{
			float radius = firstRadius + radiusStep*i_ring;

			for (int i = 0; i < ringSize; i++) {
				Point2i p(0,0);
				p.x = round( radius * cos(2 * PI * i / ringSize) );
				p.y = round( radius * sin(2 * PI * i / ringSize) );
				geometryData[i + i_ring*ringSize] = p;
			}
		}
	}

	void Descriptor::generateResults()
	{
		for(int i=0; i<17; ++i)
		{
			int value = 0;
			for( int j=0; j<i; ++j ) value += pow(2,j);
			results.push_back(std::bitset<16>(value));
		}
	}

	int Descriptor::descriptorSize() const
	{
		const int descriptor_type_size = 8/(float)numBits*ringSize*numRings;
		return descriptor_type_size;
	}

	int Descriptor::descriptorType() const
	{
		return CV_8UC1;
	}

	void Descriptor::computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		// Construct integral image for fast smoothing (box filter)
	    Mat sum;

	    Mat grayImage = image;
	    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

	    // integral( grayImage, sum, CV_32S);

	    KeyPointsFilter::runByImageBorder(keypoints, image.size(), firstRadius + radiusStep*numRings);

	    const int descriptor_type_size = 8/(float)numBits*ringSize*numRings;
    	descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);

	    for (unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp)
	    {
	        uchar* desc = descriptors.ptr(i_kp);
	        const KeyPoint& pt = keypoints[i_kp];

	        unsigned char center = valueAtCenter( pt, grayImage );
	        std::cout << (int)center << std::endl;

	        for( int i = 0; i < ringSize*numRings; ++i )
	        {
	        	unsigned char cpoint = valueAt( pt, geometryData[i], grayImage );
	        }
	    }	
	}
}