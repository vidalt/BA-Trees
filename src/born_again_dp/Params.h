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

#ifndef PARAMS_H
#define PARAMS_H

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <climits>
#include <random>

class Params
{
public:

	/* GENERAL PARAMETERS */
	int objectiveFunction;					// 0 = Depth ; 1 = NbLeaves ; 2 = Depth then NbLeaves ; 3 = NbLeaves then Depth (not yet implemented) ; 4 = Heuristic
	int nbCellsSampled;						// Number of cells sampled (for the heuristic BA trees)
	int seed;								// Random seed
	std::default_random_engine generator;	// Random number generator engine
	clock_t startTime;						// Time when the algorithm started (CPU time excluding I/O)
	clock_t stopTime;						// Time when the algorithm stopped (CPU time excluding I/O)

	/* DATASET INFORMATION */
	std::string datasetName;		// Name of the dataset
	std::string ensembleMethod;		// Name of the ensemble method used
	int nbFeatures;				    // Number of features
	int nbClasses;					// Number of classes
	int maxDepth;					// Maximum depth of the decision trees received in input
	int nbTrees;					// Number of trees in the random forest
	
	/* CONSTRUCTOR -- Reading main problem parameters from the input file */
	Params(std::ifstream & inputFile, int nbTrees, int objectiveFunction, int seed) : nbTrees(nbTrees), objectiveFunction(objectiveFunction), seed(seed)
	{
		nbCellsSampled = 1000; // Currently setting manually the number of manufactured samples and seed
		generator.seed(seed);
		std::string useless;
		inputFile >> useless >> datasetName;
		inputFile >> useless >> ensembleMethod;
		std::getline(inputFile, useless);
		if (nbTrees == -1) inputFile >> useless >> this->nbTrees;
		else inputFile >> useless >> useless;
		inputFile >> useless >> nbFeatures;
		inputFile >> useless >> nbClasses;
		inputFile >> useless >> maxDepth;
		std::getline(inputFile, useless);
		std::getline(inputFile, useless);
		std::getline(inputFile, useless);
	};
};
#endif
