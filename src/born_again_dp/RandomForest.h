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

#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "Params.h"

struct Node
{
	enum {NODE_NULL, NODE_LEAF, NODE_INTERNAL} nodeType = NODE_NULL;	// Node type
	int splitFeature = -1;											    // Split feature
	double splitValue = -1.e30;											// Split threshold (<= goes to left branch, > goes to right branch)
	int classification = -1;											// Majority class (selects the class of smallest index in case of tie)
	int nodeID = -1;                                                    // Index of the node
	int leftChild = -1;                                                 // Index of the left child
	int rightChild = -1;                                                // Index of the right child
	int depth = -1;                                                     // Depth of this node                     
};

struct LeafRegion
{
	std::vector<std::pair<int, int>> range; // A pair of vectors representing the coordinates of the region
	short int classification;	  // Class associated to this region
	LeafRegion(const std::vector<std::pair<int, int>> & range, short int classification) : range(range), classification(classification) {};
	LeafRegion() {};
};

class RandomForest
{

private:

	// Access to the problem and dataset parameters
	Params * params;

public:


	// Lists all hyperplane levels for each dimension d
	std::vector<std::vector<double>> orderedHyperplanes;

	// Random forest, represented as a vector of trees
	std::vector<std::vector <Node>> trees;

	// Leaves of the random forest represented as regions
	std::vector<std::vector<LeafRegion>> leafRegions;

	// Count vector, to optimize a bit the code
	std::vector <int> countClasses;

	// Temporary structure used in the region calculation procedure
	LeafRegion myLeaf;

	// Recursive procedure to compute the leaf regions
	void computeLeafRegions(int t, int i);

	// Returns the majority class associated to a sample
	int majorityClass(const std::vector<double> & mySample);

	// Collects all hyperplane levels
	std::vector<std::vector<double>> getHyperplanes();
	
	// Constructor
	RandomForest(Params * params, std::ifstream & inputFile);
};
#endif
