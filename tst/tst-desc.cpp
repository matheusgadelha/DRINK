/*!
 * \brief Test program for descriptor research.
 * \date March 13th, 2013.
 * \version 1.0
 * \copyright GNU Public License
 *
 * \author Matheus Gadelha, Bruno Motta
 *
 *	This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * 
 * 
 */

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;

/*!
 * Converts int to string
 * \param n Number to be converted to string
 * \returns A string =)
 */
const string toString( int n )
{
	// Creates a string stream to use on conversion
	ostringstream a_stream;
	// Pust int into string stream
	a_stream << n;

	// Transforms string stream to string
	const string result = a_stream.str();

	return result;
}

/*!
 * Main test function.
 *
 * @param argc Number of args.
 * @param argv Standard content.
 */
int main( int argc, char** argv)
{
	// Tests if was not passed the correct amount of arguments
	if( argc < 2)
	{
		cerr << "ERROR >>> Missing test argument... Use one of the following:\n";
		cerr << ". . .";
		cerr << "\tbark\n";
		cerr << "\tbikes\n";
		cerr << "\tboat\n";
		cerr << "\tgraf\n";
		cerr << "\tleuven\n";
		cerr << "\ttrees\n";
		cerr << "\twall\n";

		return -1;
	}

	// Defining test folder path
	string test_folder = argv[1];
	// Adds data folder to test name to create complete test path
	// The following defines a complete test folder path to be used at processing.
	test_folder = "data/" + test_folder + "/";

	//Creates image files prefix
	string image_prefix = test_folder + "img";

	//Creates array containing all test images
	vector<cv::Mat> images;
	for( int i=1; i <= 6; ++i )
	{
		cout << image_prefix + toString(i) + ".ppm" << std::endl;
		images.push_back(cv::Mat(image_prefix + toString(i) + ".ppm"));
	}

	cout << image_prefix << endl;

	return 0;
}