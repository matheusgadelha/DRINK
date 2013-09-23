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
unsigned char smoothedSum(
	const cv::Mat& sum,
	const cv::KeyPoint& pt, 
	int y, 
	int x, 
	const int _kernelSize
){
    static const int HALF_KERNEL = _kernelSize/2;

    int img_y = (int)(pt.pt.y) + y;
    int img_x = (int)(pt.pt.x) + x;
    int val = ( sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) /((2*HALF_KERNEL+1)*(2*HALF_KERNEL+1));
    return (unsigned char) val;
}

namespace cv{

	std::vector< int > Descriptor::result_statistics;

	Descriptor::Descriptor( int _numBits = 4, int _ringSize = 8, int _numRings = 4, int _kernelSize = 5 )
	{
		numBits = _numBits;
		ringSize =_ringSize;
		numRings = _numRings;
		kernelSize = _kernelSize;

		radiusStep = BIGGEST_RADIUS / numRings;
		firstRadius = radiusStep;

		geometryData.resize( ringSize*numRings );

		for( int i=0; i < ringSize*numRings; ++i ) geometryData[i].resize( SCALE_SAMPLES );

		for( int i=0; i < ringSize*numRings; ++i )
			for( int j=0; j < SCALE_SAMPLES; ++j )
				geometryData[i][j].resize( rotations );

		generateGeometry();
		generateResults();
		// generateRandomPairs();
	}

	void Descriptor::init( int _numBits, int _ringSize, int _numRings )
	{
		// numBits  = _numBits;
		// ringSize = _ringSize;
		// numRings = _numRings;

		// geometryData = new Point2i[ringSize*numRings];
		// generateGeometry();
		// generateResults();
	}

	void Descriptor::generateRandomPairs()
	{
		int total_pairs = ringSize * numRings * 2;

		for(int i=0; i<total_pairs; ++i)
			pairs.push_back( rand() % (ringSize*numRings) );
	}

	void Descriptor::generateGeometry()
	{
		this->smallestRadius = pow( SCALE_FACTOR, SCALE_SAMPLES-1 )*BIGGEST_RADIUS;
		float rot_angle = 360.0f/rotations;

		for( int i_ring = 0; i_ring < numRings; ++i_ring )
		{
			float radius = BIGGEST_RADIUS;

			for( int i = 0; i < ringSize; i++ ){
				Point2i p(0,0);
				p.x = round( radius * cos( 2 * PI * i / ringSize ));
				p.y = round( radius * sin( 2 * PI * i / ringSize ));
				geometryData[ i + i_ring*ringSize ][0][0] = p;
			}
		}

		for( int i_point = 0; i_point < ringSize*numRings; ++i_point )
		{
			for( int i_scale = 1; i_scale < SCALE_SAMPLES; ++i_scale )
			{
				float sclFactor = pow( SCALE_FACTOR, i_scale );
				geometryData[i_point][i_scale][0].x = sclFactor * geometryData[i_point][0][0].x;
				geometryData[i_point][i_scale][0].y = sclFactor * geometryData[i_point][0][0].y;
			}
		}

		for( int i_point = 0; i_point < ringSize*numRings; ++i_point )
		{
			for( int i_scale = 0; i_scale < SCALE_SAMPLES; ++i_scale )
			{
				for( int i_rot = 1; i_rot < rotations; ++i_rot )
				{
					geometryData[i_point][i_scale][i_rot].x = cos(rot_angle*PI/180.0f) * (float)geometryData[i_point][i_scale][i_rot-1].x +
															  sin(rot_angle*PI/180.0f) * (float)geometryData[i_point][i_scale][i_rot-1].y;

					geometryData[i_point][i_scale][i_rot].y = -sin(rot_angle*PI/180.0f) * (float)geometryData[i_point][i_scale][i_rot-1].x +
															  cos(rot_angle*PI/180.0f) * (float)geometryData[i_point][i_scale][i_rot-1].y;
				}
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

		float alpha = 0.025f;
		int disp = numBits/2;
		for( int i = -255; i <= 255; ++i )
		{
			int idx = round((disp*alpha*i)/sqrt(1+pow(alpha*i,2))) + disp;
			bins.push_back( results[ idx ] );
		}
	}

	void Descriptor::increaseStatistics( const std::bitset<RBITS> r ) const
	{
		for( unsigned i = 0; i < results.size(); ++i )
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

	void Descriptor::computeImpl(
		const Mat& image,
		std::vector<KeyPoint>& keypoints,
		Mat& descriptors ) const
	{

	    Mat sum;

	    Mat grayImage = image;
	    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

	    integral( grayImage, sum, CV_32S );

	    KeyPointsFilter::runByImageBorder(keypoints, image.size(), firstRadius + radiusStep*numRings);

	    const int descriptor_type_size = ((float)numBits/8)*ringSize*numRings;
    	descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);

    	float conversion_rot = rotations/360.0f;

	    for( unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp )
	    {
	        uchar* desc = descriptors.ptr(i_kp);
	        const KeyPoint& pt = keypoints[i_kp];

	        int pt_scale = round(log(pt.size/BIGGEST_RADIUS)/log(SCALE_FACTOR));
	        // std::cout << "Calculated scale: " << pt_scale << std::endl;
	        // std::cout << "KeyPoint size: " << pt.size << std::endl;
	        // std::cout << "---" << std::endl;
	        int rot_idx = conversion_rot * pt.angle;

	        // std::cout << pt.octave << " " << pt.size << std::endl;

	        // std::cout << "rot_idx " << rot_idx << std::endl; 

			unsigned char center = smoothedSum(
				sum,
				pt,
				0,
				0,
				kernelSize
			);

			int bit_count = 0;
			int inserted_chars = 0;

			for( int i = 0; i < geometryData.size(); i++ )
			{

				if( bit_count == 8 )
				{
					inserted_chars++;
					bit_count = 0;
				}

				// unsigned char center = smoothedSum(
				// 	sum,
				// 	pt,
				// 	geometryData[pairs[i]][pt_scale][rot_idx].x,
				// 	geometryData[pairs[i]][pt_scale][rot_idx].y,
				// 	kernelSize
				// );

				// unsigned char center = valueAt( pt, geometryData[pairs[i]][pt_scale][rot_idx], grayImage );
				// unsigned char cpoint = valueAt( pt, geometryData[pairs[i+1]][pt_scale][rot_idx], grayImage );

				unsigned char cpoint =  smoothedSum(
					sum,
					pt,
					geometryData[i][pt_scale][rot_idx].x,
					geometryData[i][pt_scale][rot_idx].y,
					kernelSize
				);

				unsigned char raw_value = 0;
				unsigned char diff = 0;

	        	if( center > cpoint ){
	        		diff = center - cpoint;
	        		raw_value = bins[255+diff].to_ulong();
	        	}
	        	else
	        	{
	        		diff = cpoint - center;
	        		raw_value = bins[255-diff].to_ulong();
	        	}

	        	increaseStatistics( raw_value );

	        	bit_count += numBits;

	        	desc[inserted_chars] += ( raw_value << (8-bit_count) );
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