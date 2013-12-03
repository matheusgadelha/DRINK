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
  const cv::Mat& img,
	const cv::Mat& sum,
	const cv::KeyPoint& pt, 
	int y, 
	int x, 
	const float _kernelSize
){
    // Interpolation code for sigma < 0.5
    // const int& imagecols = img.cols;
    // if( _kernelSize < 0.5f )
    // {
    //   // interpolation multipliers:
    //   const int r_x = static_cast<int>((int)pt.pt.x*1024);
    //   const int r_y = static_cast<int>((int)pt.pt.y*1024);
    //   const int r_x_1 = (1024-r_x);
    //   const int r_y_1 = (1024-r_y);
    //   uchar* ptr = img.data+((int)pt.pt.x+x)+((int)pt.pt.y+y)*imagecols;
    //   unsigned int ret_val;
    //   // linear interpolation:
    //   ret_val = (r_x_1*r_y_1*int(*ptr));
    //   ptr++;
    //   ret_val += (r_x*r_y_1*int(*ptr));
    //   ptr += imagecols;
    //   ret_val += (r_x*r_y*int(*ptr));
    //   ptr--;
    //   ret_val += (r_x_1*r_y*int(*ptr));
    //   //return the rounded mean
    //   ret_val += 2 * 1024 * 1024;
    //   return static_cast<uchar>(ret_val / (4 * 1024 * 1024));
    // }

    int HALF_KERNEL = std::max(_kernelSize/2,1.0f);

    int img_y = (int)(pt.pt.y) + y;
    int img_x = (int)(pt.pt.x) + x;

    // calculate borders
    const int x_left = int(img_x-HALF_KERNEL+0.5);
    const int y_top = int(img_y-HALF_KERNEL+0.5);
    const int x_right = int(img_x+HALF_KERNEL+1.5);//integral image is 1px wider
    const int y_bottom = int(img_y+HALF_KERNEL+1.5);//integral image is 1px higher
    int ret_val;

    // std::cout << "SMOOTHED " << pt.pt.x << " " << pt.pt.y << " " << x << " " << y << std::endl;

    // if( img_x > sum.cols ) std::cout << img_x << " X OUT OF RANGE\n";
    // if( img_y > sum.rows ) std::cout << pt.pt.y << " " << y << " " << img_y << " Y OUT OF RANGE\n";

    ret_val = sum.at<int>(y_bottom,x_right);//bottom right corner
    ret_val -= sum.at<int>(y_bottom,x_left);
    ret_val += sum.at<int>(y_top,x_left);
    ret_val -= sum.at<int>(y_top,x_right);
    ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );

    //~ std::cout<<sum.step[1]<<std::endl;
    return static_cast<uchar>(ret_val);

    // std::cout << "SMOTHED KERNEL " << _kernelSize/2;

    // int val = ( sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
    //        - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
    //        - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
    //        + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) /((2*HALF_KERNEL+1)*(2*HALF_KERNEL+1));
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
  std::vector< int > Descriptor::patternSizes;
	std::vector< std::vector<int> > Descriptor::pair_result_statistics;
	std::vector<Descriptor::TestPair> Descriptor::pairs;
	std::vector<Descriptor::TestPair> Descriptor::allPairsVec;
	std::vector<float> Descriptor::pair_std_dev;
	std::vector< std::vector<unsigned char> > Descriptor::data;

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

	Descriptor::TestPair::TestPair( int _a, int _b )
	{
		a = _a;
		b = _b;
		resultCount.resize(5);
	}

	Descriptor::TestPair::TestPair()
	{
		a = 0;
		b = 0;
	}

  float Descriptor::l2Distance( PatternPoint a, PatternPoint b )
  {
    return sqrt( pow(a.x-b.x,2) + pow(a.y-b.y,2) );
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

      int raw_pairs[] = {
        1553, 1595, 1596, 344, 372, 627, 375, 467, 331, 47, 303, 415, 10, 666, 289, 373, 414, 247, 89, 5, 422, 426, 215, 665, 465, 258, 299, 582, 585, 623, 667, 718, 501, 677, 657, 90, 514, 593, 551, 133, 587, 333, 877, 82, 1085, 1715, 1128, 11, 1169, 846, 550, 43, 291, 762, 719, 835, 1013, 509, 1589, 1381, 1018, 1138, 1716, 1181
       // STD SORTING
        // 302, 303, 261, 344, 262, 343, 176, 217, 56, 218, 177, 97, 98, 57, 301, 256, 135, 260, 341, 258, 338, 15, 136, 298, 12, 342, 340, 88, 214, 13, 257, 16, 299, 173, 138, 224, 47, 131, 183, 246, 300, 378, 335, 10, 194, 130, 139, 293, 259, 339, 235, 320, 373, 53, 46, 175, 179, 330, 215, 264, 95, 221, 94, 216, 140, 220, 305, 55, 379, 209, 83, 96, 0, 334, 304, 180, 190, 231, 54, 312, 263, 17, 347, 311, 142, 269, 100, 137, 11, 213, 174, 93, 248, 59, 133, 185, 354, 226, 134, 278, 14, 19, 227, 353, 270, 184, 143, 153, 22, 233, 361, 60, 204, 58, 101, 73, 244, 114, 149, 275, 30, 107, 359, 357, 156, 111, 32, 325
        // 548, 711, 1680, 1595, 1128, 922, 1213, 801, 1133, 1514, 1170, 1719, 1002, 1385, 1264, 1169, 762, 1281, 506, 1055, 256, 665, 88, 415, 299, 414, 372, 331, 247, 330, 347, 11
         // 548, 711, 921, 1680, 1595, 1638, 1553, 1717, 1596, 296, 1128, 886, 925, 930, 1214, 760, 1514, 1170, 1385, 723, 804, 1127, 1169, 1044, 762, 1302, 1223, 1086, 1427, 671, 375, 1013, 683, 1178, 506, 713, 467, 582, 539, 459, 587, 1038, 1080, 88, 415, 414, 825, 977, 47, 372, 5, 451, 331, 431, 172, 1149, 193, 1141, 305, 1191, 870, 137, 11, 1668
      };

      for( int i=0; i<sizeof(raw_pairs)/sizeof(int); ++i )
      {
        bestPairs.push_back(raw_pairs[i]);
      }

      geometryData.resize( ringSize*numRings );

      for( int i=0; i < ringSize*numRings; ++i ) 
        geometryData[i].resize( SCALE_SAMPLES );

      for( int i=0; i < ringSize*numRings; ++i )
        for( int j=0; j < SCALE_SAMPLES; ++j )
          geometryData[i][j].resize( ROTATION_SAMPLES );

      generateGeometry();
      generateAllPairs();

      if( _allPairs )
      {
        // std::cout << allPairsVec.size() << std::endl;

        // for( int i=0; i < allPairsVec.size(); ++i )
        // {
        // 	pairs.push_back( allPairsVec[i] );
        // }
        pairs = allPairsVec;
        std::cout << pairs.size() << std::endl;
      }
      else
      {
        for( unsigned i=0; i<numPairs; ++i )
        {
          pairs.push_back(allPairsVec[bestPairs[i]]);
        }
        
//        selectPairs( 100.0f, 150.0f );
        std::cout << pairs.size();
      }

      generateResults();

    }

    void Descriptor::generateRandomPairs()
    { 
      for(int i=0; i<numPairs; ++i)
      {
        pairs.push_back( TestPair(rand() % (numPoints), rand() % (numPoints)) );
      }
    }

    void Descriptor::generateAllPairs()
    {
      for( int i=0; i<numPoints; ++i )
        for( int j=0; j<numPoints; ++j )
        {
          if( j != i )
          {
            allPairsVec.push_back(TestPair(i,j));
          }
        }
    }

    void Descriptor::generateGeometry()
    {
      // PATTERN GEOMETRY
      this->smallestRadius = pow( GEOMETRY_SCALE_FACTOR, SCALE_SAMPLES-1 )*BIGGEST_RADIUS;
      this->numPoints = 0;
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
          p.sigma = (sigma_sample*2.7f);
          geometryData[ i + i_ring*ringSize ][0][0] = p;
          numPoints++;
        }
      }

      // SCALE SAMPLES
      for( int i_scale = 0; i_scale < SCALE_SAMPLES; ++i_scale )
      {
        float sclFactor = pow( SCALE_FACTOR, i_scale );
        patternSizes.push_back( ceil((BIGGEST_RADIUS+geometryData[0][0][0].sigma) * sclFactor) + 1);
      }
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

    void Descriptor::selectPairs( float _delta_min, float _delta_max )
    {
      pairs.clear();
      for( unsigned i=0; i < allPairsVec.size(); ++i )
      {
        if( l2Distance(
              geometryData[allPairsVec[i].a][0][0],
              geometryData[allPairsVec[i].b][0][0]
            ) < _delta_max && 
            l2Distance(
              geometryData[allPairsVec[i].a][0][0],
              geometryData[allPairsVec[i].b][0][0]
            ) > _delta_min
          )
        {
          this->pairs.push_back( allPairsVec[i] );
        }
        if( this->pairs.size() >= 64) break;
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
        // int idx = (5*i/510.0f) + disp;
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

    void Descriptor::increaseStatisticsForPair( const std::bitset<RBITS> r, int p, int kp ) const
    {
      for( unsigned i = 0; i < pairs[p].resultCount.size(); ++i )
      {
        if( results[i] == r)
        {
          pairs[p].resultCount[i]++;
          pairs[p].result = i;
          data[kp][p] = i;
        }
      }
    }

    int Descriptor::descriptorSize() const
    {
      const int descriptor_type_size = (numBits*pairs.size())/8;
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

      const int descriptor_type_size = (numBits*pairs.size())/8;
      const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin();

      std::vector<int> kpScales( keypoints.size() );
      const std::vector<int>::iterator scaleBegin = kpScales.begin();

      float conversion_rot = ROTATION_SAMPLES/360.0f;
      float log_scale_factor = 1/log(SCALE_FACTOR);
      float inv_biggest_radius = 1/BIGGEST_RADIUS;

      for( size_t i_kp = keypoints.size(); i_kp--; )
      {
        kpScales[i_kp] = std::max( std::min(log(keypoints[i_kp].size*inv_biggest_radius)*log_scale_factor, SCALE_SAMPLES-1.0f), 0.0f );
        if( keypoints[i_kp].pt.x <= patternSizes[kpScales[i_kp]] || //check if the description at this specific position and scale fits inside the image
            keypoints[i_kp].pt.y <= patternSizes[kpScales[i_kp]] ||
            keypoints[i_kp].pt.x >= image.cols-patternSizes[kpScales[i_kp]] ||
            keypoints[i_kp].pt.y >= image.rows-patternSizes[kpScales[i_kp]] 
          )
        {
          keypoints.erase(kpBegin+i_kp);
          kpScales.erase(scaleBegin+i_kp);
        }
      }

      descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);

      // Iterates over all keypoints
      for( unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp )
      {
        uchar* desc = descriptors.ptr(i_kp);
        const KeyPoint& pt = keypoints[i_kp];

        if( allPairs )data.push_back( std::vector<unsigned char>( 1722, 0) );

        int rot_idx = pt.angle*conversion_rot;

        int bit_count = 0;
        int inserted_chars = 0;

        for( int i = 0; i < pairs.size(); i++ )
        {

          if( bit_count == 8 )
          {    
            inserted_chars++;
            bit_count = 0;
          }

          unsigned char center;
          if(geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].sigma <= 1)
          {

            center = valueAt(
              pt,
              Point2i(geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].x,
                geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].y ),
              grayImage
            );
          }
          else
          {
            center = smoothedSum(
              grayImage,
              sum,
              pt,
              geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].x,
              geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].y,
              geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].sigma
            );
          }

          unsigned char cpoint;
          if(geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].sigma <= 1)
          {
            cpoint = valueAt(
              pt,
              Point2i(geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].x,
                geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].y ),
              grayImage
            );
          }
          else
          {
            cpoint = smoothedSum(
              grayImage,
              sum,
              pt,
              geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].x,
              geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].y,
              geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].sigma
            );
          }

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
          if( allPairs)
          {
          	increaseStatisticsForPair( raw_value, i, data.size()-1 );
          }

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