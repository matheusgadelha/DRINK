
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"
#include "FeatureAlgorithm.hpp"
#include "Descriptor.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>


const bool USE_VERBOSE_TRANSFORMATIONS = false;

int main(int argc, const char* argv[])
{
    cv::Descriptor::init();
    
    std::vector<FeatureAlgorithm>              algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    int raw_pairs[] ={
        381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
        592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
        796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
        691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
        382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
        466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
        560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
        418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
        72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
        56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
        129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
        236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
        276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
        769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
        544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
        212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
        152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
        194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
        844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
        736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
        182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
        242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
        13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
        819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
        185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
        851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
        413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
        197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
        41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
        260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
        131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
        325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
        381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
        670,249,36,581,389,605,331,518,442,822
    };

    std::vector<int> pairs;
    for(int i=1; i<= 512; ++i)
    {
        pairs.push_back(i);
    }

    bool useCrossCheck = true;

    // Initialize list of algorithm tuples:
       
    // algorithms.push_back(FeatureAlgorithm("BRISK",
    //     new cv::BRISK(60,4),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("BRIEF",
        new cv::ORB(),
        new cv::BriefDescriptorExtractor(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("DRINK B4 S8 R K5",
        new cv::ORB(),
        new cv::Descriptor(4,6,6,128,false),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("FREAK",
        new cv::ORB(),
        new cv::FREAK(true, true, 22.0f, 8 ),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("ORB",
        new cv::ORB(),
        new cv::ORB(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S16 R6 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(4,16,6,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S16 R8 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(4,16,8,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B8 S8 R5 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(8,8,5,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S8 R4 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(4,8,4,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S8 R4 K9",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(4,8,4,9),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK K7",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(7),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK K9",
    //     new cv::FastFeatureDetector(),
    //     new cv::Descriptor(9),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    
    // algorithms.push_back(FeatureAlgorithm("FREAK",
    //     new cv::SurfFeatureDetector(),
    //     new cv::FREAK(),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    /*
    algorithms.push_back(FeatureAlgorithm("SURF+BRISK",
        new cv::SurfFeatureDetector(),
        new cv::BriskDescriptorExtractor(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("SURF BF",
        new cv::SurfFeatureDetector(),
        new cv::SurfDescriptorExtractor(),
        new cv::BFMatcher(cv::NORM_L2, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("SURF FLANN",
        new cv::SurfFeatureDetector(),
        new cv::SurfDescriptorExtractor(),
        new cv::FlannBasedMatcher()));
        */


    /*
    algorithms.push_back(FeatureAlgorithm("ORB+FREAK(normalized)",
        new cv::OrbFeatureDetector(),
        new cv::FREAK(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("FREAK(normalized)",
        new cv::SurfFeatureDetector(),
        new cv::FREAK(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));


    algorithms.push_back(FeatureAlgorithm("FAST+BRIEF",
        new cv::FastFeatureDetector(50),
        new cv::BriefDescriptorExtractor(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));



    

    /**/

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS)
    {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new BrightnessImageTransform(-127, +127,1));
        transformations.push_back(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f,0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.01f));
    }
    else
    {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f,0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.1f));
        transformations.push_back(new BrightnessImageTransform(-127, +127,10));
    }

    if (argc < 2)
    {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    for (int imageIndex = 1; imageIndex < argc; imageIndex++)
    {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);

        CollectedStatistics fullStat;

        if (testImage.empty())
        {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++)
        {
            const FeatureAlgorithm& alg   = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++)
            {
                const ImageTransformation& trans = *transformations[transformIndex].obj;

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
            }

            std::cout << "done." << std::endl;
        }

        fullStat.printAverage(std::cout, StatisticsElementHomographyError);
        
        
        std::ofstream performanceStr("Performance.txt");
        fullStat.printPerformanceStatistics(performanceStr);

        std::ofstream matchingRatioStr("MatchingRatio.txt");
        fullStat.printStatistics(matchingRatioStr,  StatisticsElementMatchingRatio);

        std::ofstream percentOfMatchesStr("PercentOfMatches.txt") ;
        fullStat.printStatistics(percentOfMatchesStr, StatisticsElementPercentOfMatches);

        std::ofstream percentOfCorrectMatchesStr("PercentOfCorrectMatches.txt");
        fullStat.printStatistics(percentOfCorrectMatchesStr, StatisticsElementPercentOfCorrectMatches);

        std::ofstream meanDistanceStr("MeanDistance.txt");
        fullStat.printStatistics(meanDistanceStr, StatisticsElementMeanDistance);

        std::ofstream homographyErrorStr("HomographyError.txt");
        fullStat.printStatistics(homographyErrorStr, StatisticsElementHomographyError);

        /**/
    }

    return 0;
}

