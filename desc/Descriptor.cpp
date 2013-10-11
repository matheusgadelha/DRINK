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

inline
unsigned int k2Combination( int n )
{
	return (n * (n-1))/2;
}

inline
float stdDeviation( std::vector<int> dist, int size )
{
	float sum = 0.0f;
	float num = 0.0f;
	for( int i = 0; i < size; ++i )
	{
		sum += i*dist[i];
		num += dist[i];
	}
	// float avarage = sum / num;
	float avarage = 2;

	float sum_sqrd_diff = 0;
	for( int i = 0; i < size; ++i )
	{
		sum_sqrd_diff += pow((avarage-i),2) * dist[i];
	}

	return sqrt(sum_sqrd_diff/num);
}

namespace cv{

	std::vector< int > Descriptor::result_statistics;
	std::vector< std::vector<int> > Descriptor::pair_result_statistics;
	std::vector<float> Descriptor::pair_std_dev;

	Descriptor::PatternPoint::PatternPoint( int _x, int _y )
	{
		x = _x;
		y = _y;
		sigma = 1;
	}

	Descriptor::PatternPoint::PatternPoint()
	{
		x = 0;
		y = 0;
		sigma = 1;
	}

	Descriptor::Descriptor( int _numBits = 4, int _ringSize = 8, int _numRings = 4, int _pairs = 64, bool _allPairs = false )
	{
		numBits = _numBits;
		ringSize =_ringSize;
		numRings = _numRings;
		numPairs = _pairs;
		allPairs = _allPairs;

		radiusStep = BIGGEST_RADIUS / numRings;
		firstRadius = radiusStep;
		numPoints = 0;

		geometryData.resize( ringSize*numRings );

		for( int i=0; i < ringSize*numRings; ++i ) geometryData[i].resize( SCALE_SAMPLES );

		for( int i=0; i < ringSize*numRings; ++i )
			for( int j=0; j < SCALE_SAMPLES; ++j )
				geometryData[i][j].resize( ROTATION_SAMPLES );

		generateGeometry();

		if( _allPairs )
		{
			numPairs = k2Combination( numPoints )*2;
			generateAllPairs();
			pair_result_statistics.resize(numPairs);
		}
		else
		{
			generateRandomPairs();
		}
		generateResults();
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
		for(int i=0; i<numPairs*2; ++i)
		{
			pairs.push_back( rand() % (numPoints) );
		}
	}

	void Descriptor::generateAllPairs()
	{
		for( int i=0; i<numPoints; ++i )
			for( int j=0; j<numPoints; ++j )
			{
				if( j != i )
				{
					pairs.push_back(i);
					pairs.push_back(j);
				}
			}
	}

	void Descriptor::generateGeometry()
	{
		// PATTERN GEOMETRY
		this->smallestRadius = pow( GEOMETRY_SCALE_FACTOR, SCALE_SAMPLES-1 )*BIGGEST_RADIUS;
		float rot_angle = 360.0f/ROTATION_SAMPLES;

		float radius = BIGGEST_RADIUS;
		float inner_angle = (360.0f/((float)ringSize))/2;

		for( int i_ring = 0; i_ring < numRings; ++i_ring )
		{
			radius = pow( GEOMETRY_SCALE_FACTOR, i_ring )*BIGGEST_RADIUS;

			float sigma_sample = radius*sin(inner_angle * PI/180.0f);

			for( int i = 0; i < ringSize; i++ ){
				PatternPoint p(0,0);

				p.x = round( radius * cos( 2 * PI * i / ringSize ));
				p.y = round( radius * sin( 2 * PI * i / ringSize ));

				if( i_ring % 2 )
				{
					PatternPoint np( p.x, p.y );
					np = p;
					p.x = round( cos(PI/ringSize) * float(np.x) -
						  sin(PI/ringSize) * float(np.y) );

					p.y = round( sin(PI/ringSize) * float(np.x) +
						  cos(PI/ringSize) * float(np.y) );
				}
				p.sigma = round( sigma_sample );
				geometryData[ i + i_ring*ringSize ][0][0] = p;
				numPoints++;
			}
		}

		// SCALE SAMPLES
		for( int i_point = 0; i_point < ringSize*numRings; ++i_point )
		{
			for( int i_scale = 1; i_scale < SCALE_SAMPLES; ++i_scale )
			{
				float sclFactor = pow( SCALE_FACTOR, i_scale );
				geometryData[i_point][i_scale][0].x = sclFactor * geometryData[i_point][0][0].x;
				geometryData[i_point][i_scale][0].y = sclFactor * geometryData[i_point][0][0].y;
				geometryData[i_point][i_scale][0].sigma = sclFactor * geometryData[i_point][0][0].sigma;
			}
		}

		// ROTATION SAMPLES
		for( int i_point = 0; i_point < ringSize*numRings; ++i_point )
		{
			for( int i_scale = 0; i_scale < SCALE_SAMPLES; ++i_scale )
			{
				for( int i_rot = 1; i_rot < ROTATION_SAMPLES; ++i_rot )
				{
					geometryData[i_point][i_scale][i_rot].x = cos(rot_angle*i_rot*PI/180.0f) * (float)geometryData[i_point][i_scale][0].x +
															  sin(rot_angle*i_rot*PI/180.0f) * (float)geometryData[i_point][i_scale][0].y;

					geometryData[i_point][i_scale][i_rot].y = -sin(rot_angle*i_rot*PI/180.0f) * (float)geometryData[i_point][i_scale][0].x +
															  cos(rot_angle*i_rot*PI/180.0f) * (float)geometryData[i_point][i_scale][0].y;

					geometryData[i_point][i_scale][i_rot].sigma = geometryData[i_point][i_scale][0].sigma;
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

			for( unsigned j=0; j < pair_result_statistics.size(); ++j )
				pair_result_statistics[j].push_back(0);
		}

		float alpha = 0.015f;
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

	void Descriptor::increaseStatisticsForPair( const std::bitset<RBITS> r, int p ) const
	{
		for( unsigned i = 0; i < pair_result_statistics[p].size(); ++i )
		{
			if( results[i] == r) pair_result_statistics[p][i]++;
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

	    const int descriptor_type_size = ((float)numBits/8)*numPairs;
    	descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);

    	float conversion_rot = ROTATION_SAMPLES/360.0f;

	    for( unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp )
	    {
	        uchar* desc = descriptors.ptr(i_kp);
	        const KeyPoint& pt = keypoints[i_kp];

	        int pt_scale = round(log(pt.size/BIGGEST_RADIUS)/log(SCALE_FACTOR));
	        // int pt_scale = 3;
	        // std::cout << "Calculated scale: " << pt_scale << std::endl;
	        // std::cout << "KeyPoint size: " << pt.size << std::endl;
	        // std::cout << "---" << std::endl ;

	        int rot_idx = pt.angle*conversion_rot;
	        // int rot_idx = 0;
	        // std::cout << rot_idx << std::endl;

	        // std::cout << pt.octave << " " << pt.size << std::endl;

	        // std::cout << "rot_idx " << rot_idx << std::endl; 

			// unsigned char center = smoothedSum(
			// 	sum,
			// 	pt,
			// 	0,
			// 	0,
			// 	kernelSize
			// );

			int bit_count = 0;
			int inserted_chars = 0;

			for( int i = 0; i < numPairs; i++ )
			{

				if( bit_count == 8 )
				{
					inserted_chars++;
					bit_count = 0;
				}

				unsigned char center = smoothedSum(
					sum,
					pt,
					geometryData[pairs[i*2]][pt_scale][rot_idx].x,
					geometryData[pairs[i*2]][pt_scale][rot_idx].y,
					geometryData[pairs[i*2]][pt_scale][rot_idx].sigma
				);

				// unsigned char center = valueAt( pt, geometryData[pairs[i]][pt_scale][rot_idx], grayImage );
				// unsigned char cpoint = valueAt( pt, geometryData[pairs[i+1]][pt_scale][rot_idx], grayImage );

				unsigned char cpoint = smoothedSum(
					sum,
					pt,
					geometryData[pairs[i*2+1]][pt_scale][rot_idx].x,
					geometryData[pairs[i*2+1]][pt_scale][rot_idx].y,
					geometryData[pairs[i*2+1]][pt_scale][rot_idx].sigma
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

	        	// increaseStatistics( raw_value );
	        	if( allPairs) increaseStatisticsForPair( raw_value, i );

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

	    // std::cout <<"PAIR STAT SIZE: " << pair_result_statistics.size() << std::endl;
	    // std::cout <<"PAIR STAT PER ELEMENT SIZE: " << pair_result_statistics[1].size() << std::endl;

	    // for( int i=0; i< pair_result_statistics.size(); ++i )
	    // {
	    // 	pair_std_dev.push_back( stdDeviation(pair_result_statistics[i], numBits+1) );
	    // }

	    // for( int i=0; i < pair_std_dev.size(); ++i )
	    // {
	    // 	std::cout << pair_std_dev[i] << std::endl;
	    // }
	}
}