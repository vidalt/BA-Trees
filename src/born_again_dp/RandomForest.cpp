#include "RandomForest.h"

int RandomForest::majorityClass(const std::vector<double> & mySample)
{
	for (int c = 0; c < params->nbClasses; c++)
		countClasses[c] = 0;

	for (int t = 0; t < params->nbTrees; t++)
	{
		int i = 0; // Start at the root node and descend
		while (trees[t][i].nodeType == Node::NODE_INTERNAL)
		{
			int splitFeature = trees[t][i].splitFeature;
			double splitValue = trees[t][i].splitValue;
			if (mySample[splitFeature] <= splitValue) i = trees[t][i].leftChild;
			else i = trees[t][i].rightChild;
		}
		countClasses[trees[t][i].classification]++;
	}
	// Tie breaking criterion: in case of equality among classes, returns the class with smallest index
	return std::distance(countClasses.begin(), std::max_element(countClasses.begin(), countClasses.end()));
}

void RandomForest::computeLeafRegions(int t, int i)
{
	if (trees[t][i].nodeType == Node::NODE_LEAF)
	{
		myLeaf.classification = trees[t][i].classification;
		leafRegions[t].push_back(myLeaf);
	}
	else
	{
		int splitAttribute = trees[t][i].splitFeature;
		double splitValue = trees[t][i].splitValue;
		int splitValueIndex = std::distance(orderedHyperplanes[splitAttribute].begin(), std::find(orderedHyperplanes[splitAttribute].begin(), orderedHyperplanes[splitAttribute].end(), splitValue));

		int temp = myLeaf.range[splitAttribute].second;
		myLeaf.range[splitAttribute].second = std::min(myLeaf.range[splitAttribute].second, splitValueIndex);
		computeLeafRegions(t, trees[t][i].leftChild); // Call on left child
		myLeaf.range[splitAttribute].second = temp;

		int temp2 = myLeaf.range[splitAttribute].first;
		myLeaf.range[splitAttribute].first = std::max<int>(myLeaf.range[splitAttribute].first, splitValueIndex + 1);
		computeLeafRegions(t, trees[t][i].rightChild); // Call on right child
		myLeaf.range[splitAttribute].first = temp2;
	}
}

std::vector<std::vector<double>> RandomForest::getHyperplanes()
{
	// Collect all possible hyperplanes for all features and add them in a structure
	std::vector<std::set<double>> hyperplaneLevelsTemp = std::vector<std::set<double>>(params->nbFeatures);
	for (int t = 0; t < params->nbTrees; t++)
		for (int i = 0; i < (int)trees[t].size() ; i++)
			if (trees[t][i].nodeType == Node::NODE_INTERNAL)
				hyperplaneLevelsTemp[trees[t][i].splitFeature].insert(trees[t][i].splitValue);

	// For each feature, add a sentinel representing the maximum possible value among all features
	std::vector<std::vector<double>> myHyperplanes(params->nbFeatures);
	for (int k = 0; k < params->nbFeatures; k++)
	{
		myHyperplanes[k] = std::vector<double>(hyperplaneLevelsTemp[k].begin(), hyperplaneLevelsTemp[k].end());
		myHyperplanes[k].push_back(1.e30);
	}
	return myHyperplanes;
}

RandomForest::RandomForest(Params * params, std::ifstream & inputFile) : params(params)
{
	// Reading the input file
	std::string useless, readNodeType;
	int nbNodes;
	countClasses = std::vector <int>(params->nbClasses);
	trees = std::vector<std::vector<Node>>(params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		std::getline(inputFile, useless);
		std::cout << "READING: " << useless << std::endl;
		inputFile >> useless >> nbNodes;
		trees[t] = std::vector<Node>(nbNodes);
		for (int i = 0; i < nbNodes; i++)
		{
			inputFile >> trees[t][i].nodeID;
			inputFile >> readNodeType;
			if (readNodeType == "IN") trees[t][i].nodeType = Node::NODE_INTERNAL;
			else if (readNodeType == "LN") trees[t][i].nodeType = Node::NODE_LEAF;
			else throw std::string("ERROR: Node type non-recognized");
			inputFile >> trees[t][i].leftChild;
			inputFile >> trees[t][i].rightChild;
			inputFile >> trees[t][i].splitFeature;
			inputFile >> trees[t][i].splitValue;
			inputFile >> trees[t][i].depth;
			inputFile >> trees[t][i].classification;
		}
		std::getline(inputFile, useless);
		std::getline(inputFile, useless);
	}

	// Initializing other auxiliary structures
	leafRegions = std::vector<std::vector<LeafRegion>>(params->nbTrees);
	orderedHyperplanes = getHyperplanes();

	for (int k = 0; k < params->nbFeatures; k++)
		myLeaf.range.push_back({ 0, (int)orderedHyperplanes[k].size() - 1 });

	for (int t1 = 0; t1 < params->nbTrees; t1++)
		computeLeafRegions(t1, 0);
}