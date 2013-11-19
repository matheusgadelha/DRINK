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
        // MEAN A < B
        // 1607, 1519, 32, 247, 172, 763, 1031, 592, 1352, 1447, 305, 666, 618, 1721, 1595, 1265, 168, 859, 1514, 888, 1015, 426, 1584, 215, 1266, 648, 1101, 471, 971, 81, 1061, 1139, 1176, 792, 1332, 1624, 1386, 1349, 1131, 385, 47, 1248, 1561, 981, 498, 1551, 1710, 582, 896, 1187, 1298, 809, 242, 550, 1449, 725, 1697, 1239, 1011, 811, 935, 1070, 936, 758
        // KL SORTING A > B
        // 669, 590, 428, 926, 376, 735, 189, 696, 1184, 1587, 896, 103, 941, 743, 1692, 653, 900, 529, 860, 35, 1684, 776, 1646, 1682, 1484, 856, 1502, 1229, 816, 1019, 1294, 1542, 1011, 1588, 1391, 1056, 1424, 1721, 503, 512, 1265, 924, 624, 977, 53, 1382, 95, 557, 591, 1307, 702, 660, 1248, 1181, 1004, 209, 539, 305, 1141, 706, 1346, 1010, 1469, 426
        // KL SORTING A < B
        // 971, 888, 1043, 1050, 1380, 1176, 1427, 592, 919, 1260, 792, 1018, 1013, 666, 845, 1138, 665, 627, 214, 708, 46, 288, 289, 786, 89, 1004, 643, 660, 334, 1130, 1589, 372, 11, 1514, 1594, 1639, 1305, 1715, 130, 1556, 503, 205, 1290, 1164, 1474, 82, 1348, 298, 605, 1584, 1061, 1099, 1416, 993, 1710, 1187, 81, 824, 1617, 811, 1120, 1452, 1504, 1046
        // TVD SORTING A > B
        // 669, 711, 428, 1047, 295, 1093, 1006, 1538, 690, 1581, 777, 860, 1622, 650, 1487, 1682, 202, 121, 1607, 1482, 1684, 38, 245, 1537, 736, 775, 1661, 1577, 1188, 1219, 1019, 1145, 1229, 1542, 935, 1268, 791, 631, 1632, 887, 1365, 1595, 1337, 1381, 1079, 576, 53, 870, 1433, 1417, 341, 598, 1336, 305, 420, 1462, 293, 373, 1346, 173, 1131, 677, 1297, 1260
        // TVD SORTING A < B 
      // 1169, 1212, 762, 1176, 1091, 1139, 1427, 592, 919, 792, 1260, 1459, 1254, 845, 677, 1339, 384, 627, 214, 173, 467, 88, 47, 258, 431, 1004, 969, 786, 643, 89, 1514, 131, 1513, 1639, 257, 10, 1305, 1589, 623, 1434, 1556, 1290, 0, 1456, 184, 227, 142, 605, 1348, 1099, 935, 392, 831, 1103, 1542, 1668, 1710, 811, 1452, 1120, 1166, 1498, 91, 1046
        // STD SORTING 5x4
        // 286, 16, 1, 287, 364, 57, 94, 39, 79, 0, 290, 128, 149, 3, 78, 40, 370, 58, 178, 119, 115, 169, 77, 160, 156, 297, 4, 155, 116, 206, 140, 101, 374, 341, 260, 139, 239, 224, 300, 223, 103, 208, 295, 283, 238, 301, 360, 335, 67, 284, 122, 86, 336, 182, 337, 157, 112, 225, 236, 202, 158, 126, 376, 245
        // 277, 258, 318, 226, 278, 217, 127, 261, 377, 218, 240, 108, 121, 163, 320, 246, 340, 86, 216, 200, 284, 207, 212, 267, 379, 201, 283, 204, 264, 243, 51, 300, 224, 273, 199, 341, 184, 265, 130, 214, 209, 138, 266, 165, 215, 333, 116, 155, 164, 297, 197, 83, 179, 24, 100, 44, 21, 96, 40, 175, 136, 233, 76, 0
       // STD SORTING
        // 548, 711, 1680, 1595, 1128, 922, 1213, 801, 1133, 1514, 1170, 1719, 1002, 1385, 1264, 1169, 762, 1281, 506, 1055, 256, 665, 88, 415, 299, 414, 372, 331, 247, 330, 347, 11
         548, 711, 921, 1680, 1595, 1638, 1553, 1717, 1596, 296, 1128, 886, 925, 930, 1214, 760, 1514, 1170, 1385, 723, 804, 1127, 1169, 1044, 762, 1302, 1223, 1086, 1427, 671, 375, 1013, 683, 1178, 506, 713, 467, 582, 539, 459, 587, 1038, 1080, 88, 415, 414, 825, 977, 47, 372, 5, 451, 331, 431, 172, 1149, 193, 1141, 305, 1191, 870, 137, 11, 1668
       // NO SORTING
      // 0, 2, 11, 42, 53, 85, 89, 131, 168, 172, 210, 221, 252, 263, 300, 337, 347, 379, 398, 425, 426, 462, 504, 515, 551, 588, 599, 630, 641, 674, 683, 717, 722, 757, 758, 762, 804, 842, 855, 893, 896, 924, 926, 929, 977, 1009, 1010, 1011, 1055, 1102, 1135, 1143, 1145, 1187, 1219, 1222, 1261, 1308, 1346, 1349, 1386, 1387, 1391, 1431
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
          p.sigma = (sigma_sample*0.7f);
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

      //KeyPointsFilter::runByImageBorder(keypoints, image.size(), BIGGEST_RADIUS+geometryData[0][0][0].sigma);

      const int descriptor_type_size = (numBits*pairs.size())/8;
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

        if( allPairs )data.push_back( std::vector<unsigned char>( 1722, 0) );

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