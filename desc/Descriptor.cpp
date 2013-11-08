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
    const int& imagecols = img.cols;
    if( _kernelSize < 0.5f )
    {
      // interpolation multipliers:
      const int r_x = static_cast<int>((int)pt.pt.x*1024);
      const int r_y = static_cast<int>((int)pt.pt.y*1024);
      const int r_x_1 = (1024-r_x);
      const int r_y_1 = (1024-r_y);
      uchar* ptr = img.data+((int)pt.pt.x+x)+((int)pt.pt.y+y)*imagecols;
      unsigned int ret_val;
      // linear interpolation:
      ret_val = (r_x_1*r_y_1*int(*ptr));
      ptr++;
      ret_val += (r_x*r_y_1*int(*ptr));
      ptr += imagecols;
      ret_val += (r_x*r_y*int(*ptr));
      ptr--;
      ret_val += (r_x_1*r_y*int(*ptr));
      //return the rounded mean
      ret_val += 2 * 1024 * 1024;
      return static_cast<uchar>(ret_val / (4 * 1024 * 1024));
    }

    static const int HALF_KERNEL = (_kernelSize+0.5)/2;

    int img_y = (int)(pt.pt.y) + y;
    int img_x = (int)(pt.pt.x) + x;
    int val = ( sum.at<int>(img_y + HALF_KERNEL + 1, img_x + HALF_KERNEL + 1)
           - sum.at<int>(img_y + HALF_KERNEL + 1, img_x - HALF_KERNEL)
           - sum.at<int>(img_y - HALF_KERNEL, img_x + HALF_KERNEL + 1)
           + sum.at<int>(img_y - HALF_KERNEL, img_x - HALF_KERNEL)) /((2*HALF_KERNEL+1)*(2*HALF_KERNEL+1));

    return (unsigned char)val;
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
      0, 2, 11, 42, 53, 85, 89, 131, 168, 172, 210, 221, 252, 263, 300, 337, 347, 379, 398, 425, 426, 462, 504, 515, 551, 588, 599, 630, 641, 674, 683, 717, 722, 757, 758, 762, 804, 842, 855, 893, 896, 924, 926, 929, 977, 1009, 1010, 1011, 1055, 1102, 1135, 1143, 1145, 1187, 1219, 1222, 1261, 1308, 1346, 1349, 1386, 1387, 1391, 1431
     //    0,  1,  2,  3,  4,  5, 10, 11, 18,
     //   19, 20, 38, 42, 43, 44, 46, 47,
     //   53, 59, 60, 73, 84, 85, 88, 89,
     //   95, 99, 100, 104, 126, 127, 130, 131,
     //  137, 140, 144, 145, 158, 168, 172, 173,
     //  179, 181, 185, 186, 199, 210, 211, 214,
     //  215, 221, 224, 225, 226, 235, 243, 252,
     //  253, 254, 257, 258, 263, 268, 271, 281,
     //  294, 295, 298, 299, 300, 305, 316, 320,
     //  336, 337, 338, 339, 341, 342, 347, 352,
     //  360, 378, 379, 383, 384, 389, 392, 393,
     //  397, 420, 421, 422, 424, 425, 426, 431,
     //  435, 442, 462, 463, 465, 467, 473, 476, 
     //  478, 489, 506, 509, 514, 515, 517, 519,
     //  546, 547, 550, 551, 557, 560, 564, 588,
     //  589, 592, 593, 597, 599, 602, 603, 634,
     //  635, 641, 643, 644, 645, 672, 676, 677,
     //  679, 680, 683, 690, 693, 714, 715, 718,
     //  719, 725, 729, 732, 733, 756, 757, 758,
     //  761, 762, 763, 764, 767, 769, 771, 798,
     //  799, 803, 804, 809, 812, 814, 840, 841,
     //  842, 843, 845, 846, 849, 851, 860, 882,
     //  883, 887, 888, 889, 893, 895, 896, 897,
     //  924, 928, 929, 930, 931, 932, 935, 966,
     //  967, 969, 971, 974, 977, 982, 1008, 1010,
     // 1012, 1013, 1015, 1016, 1019, 1021, 1022, 1050,
     // 1054, 1055, 1057, 1058, 1061, 1092, 1093, 1094,
     // 1096, 1097, 1103, 1136, 1138, 1139, 1141, 1142,
     // 1145, 1147, 1176, 1179, 1180, 1181, 1183, 1187,
     // 1220, 1222, 1223, 1227, 1229, 1260, 1261, 1262,
     // 1265, 1270, 1302, 1305, 1307, 1308, 1310, 1346,
     // 1347, 1350, 1386, 1387, 1390, 1391, 1392
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
        std::cout << allPairsVec.size() << std::endl;

        // for( int i=0; i < allPairsVec.size(); ++i )
        // {
        // 	pairs.push_back( allPairsVec[i] );
        // }
        pairs = allPairsVec;
        for( unsigned i=0; i < pairs.size(); ++i )
        {
          std::cout << l2Distance( 
            geometryData[pairs[i].a][0][0],
            geometryData[pairs[i].b][0][0] 
          );
        }

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
          p.sigma = sigma_sample*0.7f;
          geometryData[ i + i_ring*ringSize ][0][0] = p;
          numPoints++;
        }
      }

      // SCALE SAMPLES
      for( int i_scale = 0; i_scale < SCALE_SAMPLES; ++i_scale )
      {
        float sclFactor = pow( SCALE_FACTOR, i_scale );
        patternSizes.push_back(BIGGEST_RADIUS * sclFactor);
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

      float alpha = 0.075f;
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

      //KeyPointsFilter::runByImageBorder(keypoints, image.size(), BIGGEST_RADIUS+geometryData[0][0][0].sigma);

      const int descriptor_type_size = ((float)numBits/8)*pairs.size();
      descriptors = Mat::zeros((int)keypoints.size(), descriptor_type_size, CV_8U);
      const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin();

      std::vector<int> kpScales( keypoints.size() );
      const std::vector<int>::iterator scaleBegin = kpScales.begin();

      float conversion_rot = ROTATION_SAMPLES/360.0f;
      float log_scale_factor = 1/log(SCALE_FACTOR);
      float inv_biggest_radius = 1/BIGGEST_RADIUS;

      for( size_t i_kp = 0; i_kp < keypoints.size(); ++i_kp )
      {
        kpScales[i_kp] = round(log(keypoints[i_kp].size*inv_biggest_radius)*log_scale_factor);
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

      // Iterates over all keypoints
      for( unsigned int i_kp = 0; i_kp < keypoints.size(); ++i_kp )
      {
        uchar* desc = descriptors.ptr(i_kp);
        const KeyPoint& pt = keypoints[i_kp];

        if( allPairs )data.push_back( std::vector<unsigned char>(1722,0) );

        // int pt_scale = 0;
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

        for( int i = 0; i < pairs.size(); i++ )
        {

          if( bit_count == 8 )
          {
            inserted_chars++;
            bit_count = 0;
          }

          unsigned char center;
          if(geometryData[pairs[i].a][kpScales[i_kp]][rot_idx].sigma < 1)
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

          // unsigned char center = valueAt(
          // 	pt,
          // 	Point2i( geometryData[pairs[i*2]][pt_scale][rot_idx].x, geometryData[pairs[i*2]][pt_scale][rot_idx].y ),
          // 	grayImage
          // );

          // unsigned char cpoint = valueAt(
          // 	pt,
          // 	Point2i( geometryData[pairs[i*2+1]][pt_scale][rot_idx].x, geometryData[pairs[i*2+1]][pt_scale][rot_idx].y ),
          // 	grayImage
          // );

          unsigned char cpoint;
          if(geometryData[pairs[i].b][kpScales[i_kp]][rot_idx].sigma < 1)
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
          // if( allPairs)
          // {
          // 	increaseStatisticsForPair( raw_value, i, data.size()-1 );
          // }

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