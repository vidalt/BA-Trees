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

#ifndef BORNAGAINDECISIONTREE_H
#define BORNAGAINDECISIONTREE_H

#include "Params.h"
#include "RandomForest.h"
#include "FSpace.h"
#include <set>

#ifdef USING_CPLEX
#include "MIPCertificate.h"
#endif

#define BIG_M 100000 // Weight attributed to the depth in objective 2 (optimize weight then number of leaves)

class BornAgainDecisionTree
{
private:
	
	Params * params;				                                    // Access to the problem and dataset parameters
	RandomForest * randomForest;	                                    // Access to the random forest serving as input
	FSpace fspaceOriginal;									            // Original feature space
	FSpace fspaceFinal;										            // Feature space after filtering unnecessary hyperplanes

#ifdef USING_CPLEX
	MIPCertificate * myMIPcertificate;									// Access to the solver
#endif

	// Variables used by the heuristic BA trees
	std::vector<std::vector<double>> orderedHyperplaneLevelsHeuristic;  // Ordered hyperplane values for each feature
	std::vector<unsigned short int> bottomLeftCell;                     // Bottom left cell (for the heuristic)
	std::vector<unsigned short int> topRightCell;			            // Top right cell (for the heuristic)
	std::vector<std::vector<unsigned short int>> sampledCellsIndices;   // vector of cell indices representing the samples (for the heuristic)
	std::vector<std::vector<double>> sampledCellsCoords;				// vector of cell coordinates representing the samples (for the heuristic)
	std::vector<unsigned short int> classSampledCells;                  // class of the sampled cells
	std::set<int> nonTrivialFeatures;                                   // Contains the list of features which are not restricted to a single index
	std::set<int> nonTrivialFeaturesBeforeSplits;                       // Contains the list of features which are not restricted to a single index (not updated when splitting)

	// The regions are maintained contiguously in a vector using a perfect hash
	// This demonstrated a better memory performance than an unordered_map for obj 1 and 2
	std::vector<std::vector<unsigned int>> regions;

	// Born-again tree produced by the algorithm, using the same internal representation as scikit-learn
	std::vector<Node> rebornTree;

	// Run statistics
	unsigned long int iterationsDP;
	unsigned long int regionsMemorizedDP;
	unsigned int finalObjective;
	unsigned int finalSplits;
	unsigned int finalLeaves;
	unsigned int finalDepth;
	
	// Dynamic programming procedure to reconstruct the decision tree -- optimizing the depth of the tree (D)
	unsigned int dynamicProgrammingOptimizeDepth(int indexBottom, int indexTop);

	// Dynamic programming procedure to reconstruct the decision tree -- optimizing the number of splits in the tree -- equivalent to optimizing the number of leaves (L)
	unsigned int dynamicProgrammingOptimizeNbSplits(int indexBottom, int indexTop);

	// Dynamic programming procedure to reconstruct the decision tree -- optimizing the depth and then the number of splits as a secondary objective (DL)
	unsigned int dynamicProgrammingOptimizeDepthThenNbSplits(int indexBottom, int indexTop);

	// Recursive procedure for the heuristic construction of the decision tree
	int recursiveHelperHeuristic(unsigned int currentDepth);

	// Final procedure to extract the final solution from the DP memory
	int collectResultDP(int indexBottom, int indexTop, unsigned int optValue, unsigned int currentDepth);

public:

    // Main procedure: Building the reborn decision tree (using an exact algorithm) -- the result is guaranteed to have the smallest size and to be faithful
    void buildOptimal();

	// Main procedure: Building the reborn decision tree (using an heuristic based on data manufacturing + an oracle) -- the result is still guaranteed to be faithful
	void buildHeuristic();

	// Displays some statistics about the run
	void displayRunStatistics();

	// Exports some statistics about the run in a file
	void exportRunStatistics(std::string fileName);

	// Exports the born-again tree in a file
	void exportBATree(std::string fileName);

	// Constructor
	BornAgainDecisionTree(Params * params, RandomForest * randomForest): params(params), randomForest(randomForest), fspaceOriginal(params, randomForest), fspaceFinal(params, randomForest){};
};

#endif