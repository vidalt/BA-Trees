/*MIT License

Copyright(c) 2020 Thibaut Vidal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef MIPCERTIFICATE_H
#define MIPCERTIFICATE_H

#include "Params.h"
#include "RandomForest.h"
#define ILOSTLBEGIN
#include <ilcplex/ilocplex.h>
using namespace std;

class MIPCertificate
{
private:
	
	// Access to the problem and dataset parameters
	Params * params;

	// Access to the random forest serving as input
	RandomForest * randomForest;

public:

	// Function used to make an exact check to know if the region is pure
    // Returns TRUE if the region is guaranteed to be PURE
    // Returns FALSE otherwise
	bool buildAndRunMIP(const std::set<int> & nonTrivialFeaturesBeforeSplits, const std::set<int> & nonTrivialFeatures, const std::vector<std::vector<double>> & orderedHyperplanes, const std::vector<unsigned short int> & bottomLeftCell, const std::vector<unsigned short int> & topRightCell, int pureClass, int otherClass); // Runs the MIP

	// Function used to make a quick check to filter cases where the region is obviously pure
	// Returns TRUE if the region is guaranteed to be PURE
	// Returns FALSE if the analysis is inconclusive
	bool preFilterMinMax(const std::set<int> & nonTrivialFeaturesBeforeSplits, const std::vector<std::vector<double>> & orderedHyperplanes, const std::vector<unsigned short int> & bottomLeftCell, const std::vector<unsigned short int> & topRightCell, int pureClass, int otherClass);

	// Constructor
	MIPCertificate(Params * params, RandomForest * randomForest): params(params), randomForest(randomForest)
	{};
};

#endif