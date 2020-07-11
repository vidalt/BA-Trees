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

#ifndef FSPACE_H
#define FSPACE_H

#include "Params.h"
#include "RandomForest.h"

class FSpace
{
public:

	Params * params;                                             // Access to the problem and dataset parameters
	RandomForest * randomForest;                                 // Access to the random forest serving as input
	std::vector<std::vector<double>> orderedHyperplaneLevels;    // Lists all hyperplane levels
	std::vector<int> codeBook;                                   // Codebook used to convert cells into indices
	std::vector<int> cells;                                      // Class of each cell. Indexed based on the codeBook
	long int nbCells;                                            // Number of cells
	double nbPossibleRegions;                                    // Bound on the number of possible DP states

	// Auxiliary variables for some recursive functions
	std::vector<double> myCellValues;
	bool detectedDifference;
	int hypFeature, hypLevel;
	
	// Converts a cell to its associated hash code
	int cellToKey(const std::vector<int> & myCell);

	// Gets the kth value of the cell for a given hash code
	int keyToCell(int key, int k);

	// Generates the hash code for a region (useful for perfect hashing)
	int keyToHash(int keyBottomLeft, int keyTopRight);

	// Evaluates the majority class for each cell
	void enumerateCellsRecursion(int k, int myCellIndex);

	// Helper function to filter unnecessary hyperplanes
	void hyperplaneFilteringRecursion(int k, int myCellIndex);

	// Initializes the cells based on a list of hyperplanes
	void initializeCells(const std::vector<std::vector<double>> & hyperplanes, bool isFiltered);

	// Exports only the useful hyperplanes
	std::vector<std::vector<double>> exportUsefulHyperplanes();

	// Constructor
	FSpace(Params * params, RandomForest * randomForest) : params(params), randomForest(randomForest)
	{
		orderedHyperplaneLevels = std::vector<std::vector<double>>(params->nbFeatures);
		codeBook = std::vector<int>(params->nbFeatures, 1);
		myCellValues = std::vector<double>(params->nbFeatures);
	}
};

#endif