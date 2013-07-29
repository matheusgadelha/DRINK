#include "Descriptor.hpp"

namespace cv{

	int Descriptor::numBits;
	int Descriptor::ringSize;
	int Descriptor::numRings;

	const int Descriptor::firstRadius;
	const int Descriptor::radiusStep;

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
		
	}
}