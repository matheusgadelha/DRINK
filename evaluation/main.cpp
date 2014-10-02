
#include "ImageTransformation.hpp"
#include "AlgorithmEstimation.hpp"
#include "FeatureAlgorithm.hpp"
#include "DRINK.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

string itos(int i) // convert int to string
{
    stringstream s;
    s << i;
    return s.str();
}

const bool USE_VERBOSE_TRANSFORMATIONS = false;

int main(int argc, const char* argv[]) {
    //cv::DRINK::init();

    std::vector<FeatureAlgorithm> algorithms;
    std::vector<cv::Ptr<ImageTransformation> > transformations;

    bool useCrossCheck = true;

    // Initialize list of algorithm tuples:

    // algorithms.push_back(FeatureAlgorithm("BRISK",
    //     new cv::BRISK(60,4),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("SURF",
    //     new cv::SURF(2000,4),
    //     new cv::SURF(),
    //     new cv::BFMatcher(cv::NORM_L2, useCrossCheck)));

    // for( int i=4; i<15; ++i )
    //     for( int j=7; j<9; ++j)
    //     {
    //         algorithms.push_back(FeatureAlgorithm("DRINK S"+itos(i)+" R"+itos(j)),
    //             new cv::ORB(),
    //             new cv::DRINK(4,i,j,64,false),
    //             new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));            
    //     }

    algorithms.push_back(FeatureAlgorithm("DRINK",
            new cv::ORB(),
            new cv::DRINK(4, true, 6, 7, 64, false),
            new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("FREAK",
            new cv::ORB(),
            new cv::FREAK(true, true),
            new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("ORB",
            new cv::ORB(),
            new cv::ORB(),
            new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S16 R6 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(4,16,6,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S16 R8 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(4,16,8,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B8 S8 R5 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(8,8,5,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S8 R4 K5",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(4,8,4,5),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK B4 S8 R4 K9",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(4,8,4,9),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK K7",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(7),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    // algorithms.push_back(FeatureAlgorithm("DRINK K9",
    //     new cv::FastFeatureDetector(),
    //     new cv::DRINK(9),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));


    // algorithms.push_back(FeatureAlgorithm("FREAK",
    //     new cv::SurfFeatureDetector(),
    //     new cv::FREAK(),
    //     new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    /*
    algorithms.push_back(FeatureAlgorithm("SURF+BRISK",
        new cv::SurfFeatureDetector(),
        new cv::BriskDRINKExtractor(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("SURF BF",
        new cv::SurfFeatureDetector(),
        new cv::SurfDRINKExtractor(),
        new cv::BFMatcher(cv::NORM_L2, useCrossCheck)));

    algorithms.push_back(FeatureAlgorithm("SURF FLANN",
        new cv::SurfFeatureDetector(),
        new cv::SurfDRINKExtractor(),
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
        new cv::BriefDRINKExtractor(),
        new cv::BFMatcher(cv::NORM_HAMMING, useCrossCheck)));
     */

    // Initialize list of used transformations:
    if (USE_VERBOSE_TRANSFORMATIONS) {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new BrightnessImageTransform(-127, +127, 1));
        transformations.push_back(new ImageRotationTransformation(0, 360, 1, cv::Point2f(0.5f, 0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.01f));
    } else {
        transformations.push_back(new GaussianBlurTransform(9));
        transformations.push_back(new ImageRotationTransformation(0, 360, 10, cv::Point2f(0.5f, 0.5f)));
        transformations.push_back(new ImageScalingTransformation(0.25f, 2.0f, 0.1f));
        transformations.push_back(new BrightnessImageTransform(-127, +127, 10));
        // transformations.push_back(new YRotationTransform(-90, 90.0f, 1.0f));
        // transformations.push_back(new XRotationTransform(-90, 90.0f, 1.0f));
    }

    if (argc < 2) {
        std::cout << "At least one input image should be passed" << std::endl;
    }

    for (int imageIndex = 1; imageIndex < argc; imageIndex++) {
        std::string testImagePath(argv[imageIndex]);
        cv::Mat testImage = cv::imread(testImagePath);

        CollectedStatistics fullStat;

        if (testImage.empty()) {
            std::cout << "Cannot read image from " << testImagePath << std::endl;
        }

        for (size_t algIndex = 0; algIndex < algorithms.size(); algIndex++) {
            const FeatureAlgorithm& alg = algorithms[algIndex];

            std::cout << "Testing " << alg.name << "...";

            for (size_t transformIndex = 0; transformIndex < transformations.size(); transformIndex++) {
                const ImageTransformation& trans = *transformations[transformIndex].obj;

                performEstimation(alg, trans, testImage.clone(), fullStat.getStatistics(alg.name, trans.name));
            }

            std::cout << "done." << std::endl;
        }

        fullStat.printAverage(std::cout, StatisticsElementHomographyError);

        std::ofstream performanceStr("Performance.txt");
        fullStat.printPerformanceStatistics(performanceStr);

        std::ofstream matchingRatioStr("MatchingRatio.txt");
        fullStat.printStatistics(matchingRatioStr, StatisticsElementMatchingRatio);

        std::ofstream percentOfMatchesStr("PercentOfMatches.txt");
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

