#include "Descriptor.hpp"

namespace cv{

	int Descriptor::numBits;
	int Descriptor::ringSize;
	int Descriptor::numRings;

	const int Descriptor::firstRadius;
	const int Descriptor::radiusStep;
	const int Descriptor::kernelSize;

	Point2i* Descriptor::geometryData;

	void Descriptor::init( int _numBits, int _ringSize, int _numRings )
	{
		numBits  = _numBits;
		ringSize = _ringSize;
		numRings = _numRings;

		geometryData = new Point2i[ringSize*numRings];
		generateGeometry();
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

	void Descriptor::computeImpl(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		// Construct integral image for fast smoothing (box filter)
	    Mat sum;

	    Mat grayImage = image;
	    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

	    // integral( grayImage, sum, CV_32S);

	    KeyPointsFilter::runByImageBorder(keypoints, image.size(), firstRadius + radiusStep*numRings);

	    const int descriptor
    	descriptors = Mat::zeros((int)keypoints.size(), numBits*ringSize*numRings, CV_8U);

	    for (int i_kp = 0; i_kp < (int)keypoints.size(); ++i_kp)
	    {
	        uchar* desc = descriptors.ptr(i_kp);
	        const KeyPoint& pt = keypoints[i_kp];

	        for( int i = 0; i < keypoints.size(); ++i )
	        {

	        }
	    }	
	}
}