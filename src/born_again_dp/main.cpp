#include "Commandline.h"
#include "Params.h"
#include "RandomForest.h"
#include "BornAgainDecisionTree.h"

int main(int argc, char** argv)
{
	Commandline c(argc, argv);
	if (c.command_ok)
	{
		std::ifstream inputFile(c.instance_name.c_str());
		if (inputFile.is_open())
		{
			/* READING INPUT RANDOM FOREST */
			std::cout << "----- READING RANDOM FOREST from " << c.instance_name << std::endl;
			Params params(inputFile, c.nbTrees, c.objectiveFunction,c.seed);
			RandomForest randomForest(&params, inputFile);

			/* CONSTRUCTING THE BORN-AGAIN TREE */
			params.startTime = clock();
			BornAgainDecisionTree bornAgainTree(&params, &randomForest);
			if (c.objectiveFunction == 0 || c.objectiveFunction == 1 || c.objectiveFunction == 2)
				bornAgainTree.buildOptimal();
			else if (c.objectiveFunction == 4)
				bornAgainTree.buildHeuristic();
			params.stopTime = clock();

			/* EXPORTING STATISTICS AND RESULTS */
			bornAgainTree.displayRunStatistics();
			bornAgainTree.exportRunStatistics(c.output_name + ".out");
			bornAgainTree.exportBATree(c.output_name + ".tree");
		}
		else
		{
			std::cout << "----- IMPOSSIBLE TO READ RANDOM FOREST from: " << c.instance_name << std::endl;
		}
	}
	else
	{
		std::cout << "----- INCORRECT COMMANDLINE " << std::endl;
	}
	std::cout << "----- END OF ALGORITHM " << std::endl << std::endl;
	return 0;
}
