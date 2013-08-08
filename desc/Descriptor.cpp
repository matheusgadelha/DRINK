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

inline
unsigned char smoothedSum(const cv::Mat& sum, const cv::KeyPoint& pt, int y, int x)
{
    static const int HALF_KERNEL = cv::Descriptor::kernelSize/2;

    int img_y = (int)(pt.pt.y) + y;
    int img_x = (int)(pt.pt.x) + x;
    int val = ( sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) /((2*HALF_KERNEL+1)*(2*HALF_KERNEL+1));
    return (unsigned char) val;
}

namespace cv{

	int Descriptor::numBits;
	int Descriptor::ringSize;
	int Descriptor::numRings;

	const int Descriptor::firstRadius;
	const int Descriptor::radiusStep;
	int Descriptor::kernelSize;

	Point2i* Descriptor::geometryData;
	std::vector< std::bitset<16> > Descriptor::results;
	std::vector< std::bitset<16> > Descriptor::positiveBin;
	std::vector< std::bitset<16> > Descriptor::negativeBin;
	std::vector< int > Descriptor::result_statistics;

	Descriptor::Descriptor( int _kernelSize = 7 )
	{
		kernelSize = _kernelSize;
	}

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
		for(int i=0; i<RBITS+1; ++i)
		{
			int value = 0;
			for( int j=0; j<i; ++j ) value += pow(2,j);
			results.push_back(std::bitset<RBITS>(value));
			result_statistics.push_back(0);
		}

		float t = 127 / numBits;

		for( int i=0; i<126; ++i )
		{
			// std::cout << (i/(int)t);
			positiveBin.push_back( results[ (i/(int)t) + (numBits/2) ] );
			negativeBin.push_back( results[ (numBits/2) - (i/(int)t) ] );
		}
	}

	void Descriptor::increaseStatistics( std::bitset<RBITS> r )
	{
		for( int i = 0; i < results.size(); ++i )
		{
			if( results[i] == r) result_statistics[i]++;
		}
	}

	int Descriptor::descriptorSize() const
	{
		const int descriptor_type_size = ((float)numBits/8)*ringSize*numRings;
		return descriptor_type_size;
	}

	int Descriptor::descriptorType() const
	{
		return CV_8UC1;
	}

	void Descriptor::computeImpl( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const
	{
		// Construct integral image for fast smoothing (box filter)
	    Mat sum;

	    Mat grayImage = image;
	    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

	    integral( grayImage, sum, CV_32S);

	    KeyPointsFilter::runByImageBorder(keypoints, image.size(), firstRadius + radiusStep*numRings);

	    const int descriptor_type_size = ((float)numBits/8)*ringSize*numRings;
    	descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);
    	
    	const unsigned char r_possibilities = numBits+1;
    	const unsigned char step = 510/r_possibilities;

    	int byte_pos = 0;

	    for (unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp)
	    {
	        uchar* desc = descriptors.ptr(i_kp);
	        const KeyPoint& pt = keypoints[i_kp];

	        // unsigned char center = valueAtCenter( pt, grayImage );
	        unsigned char center = smoothedSum( sum, pt, 0, 0 );

	        // std::cout << "Center: " << (int)center << std::endl << "Smoothed: " << (int)center_smoothed << std::endl;

	        int bit_count = 0;
	        int inserted_chars = 0;
	        unsigned char val = 0;

	        for( int i = 0; i < ringSize*numRings; ++i )
	        {
	        	if( bit_count == 8 )
	        	{
	        		// std::cout << std::bitset<8>(desc[inserted_chars]) << std::endl;
	        		inserted_chars++;
	        		bit_count = 0;
	        	}

	        	// unsigned char cpoint = valueAt( pt, geometryData[i], grayImage );
	        	unsigned char cpoint = smoothedSum( sum, pt, geometryData[i].y, geometryData[i].x );
	        	unsigned char raw_value = 0;
	        	unsigned char diff = 0;

	        	if( center > cpoint ){ //center - cpoint is positive
	        		diff = center - cpoint;
	        		raw_value = positiveBin[diff].to_ulong();
	        		increaseStatistics( positiveBin[diff] );
	        		// result_statistics[r_possibilities/2 + diff/step]++;
	        		// std::cout << std::bitset<4>(raw_value) << std::endl;
	        	}
	        	else // center - cpoint is negative
	        	{
	        		diff = cpoint - center;
	        		raw_value = negativeBin[diff].to_ulong();
	        		increaseStatistics( negativeBin[diff] );
	        		// result_statistics[r_possibilities/2 - diff/step]++;
	        		// std::cout << std::bitset<4>(raw_value) << std::endl;
	        	}

	        	bit_count += numBits;

	        	desc[inserted_chars] += (raw_value << (8-bit_count));
	        }

	        // std::cout << "INSERTED CHARS:" << inserted_chars+1 << std::endl;

	        // for( int i=0; i < descriptor_type_size; ++i )
	        // {
	        // 	std::cout << std::bitset<8>(desc[i]) << " ";
	        // 	// std::cout << (int)desc[i] << " ";
	        // }
	        // std::cout << std::endl;
	    }	
	}
}