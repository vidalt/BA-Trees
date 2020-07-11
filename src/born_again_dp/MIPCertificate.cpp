#include "MIPCertificate.h"

bool MIPCertificate::buildAndRunMIP(const std::set<int> & nonTrivialFeaturesBeforeSplits, const std::set<int> & nonTrivialFeatures, const std::vector<std::vector<double>> & orderedHyperplanes, const std::vector<unsigned short int> & bottomLeftCell, const std::vector<unsigned short int> & topRightCell, int pureClass, int otherClass)
{
	IloEnv env; // CPLEX environment
	IloModel model = IloModel(env); // CPLEX model
	stringstream name;
	IloExpr expr(env);
	
	// Tracking the features which are non trivial
	int nbNonTrivialFeatures = nonTrivialFeatures.size();
	vector<int> vectorNonTrivialFeatures;
	for (int d : nonTrivialFeatures) 
		vectorNonTrivialFeatures.push_back(d);

	// Variables
	IloArray<IloNumVarArray> x;                 	            // x[t][i] = 1 if leaf i of tree t is selected for node a
	IloArray<IloArray<IloNumVarArray>> z;                       // z[t][i][d] represents the index of the cell which is selected if leaf i of tree t is selected, 0 otherwise for node a
	IloNumVarArray zz;                                          // zz[d] represents the index of the cell which is selected and common to all trees for node a

	// Constraints
	IloRangeArray constraints_selection_leaf;									// Constraints (1)  -- Select one leaf per tree: sum(i, x[t][i]) == 1  for all t            
	IloArray<IloArray<IloRangeArray>> constraints_range_positionL;			    // Constraints (2L) -- Reduces the range of the position depending on the leaf which is selected -- x[t][i] * rangeLow_{tid} <= z[t][i][d] for all i, t, d
	IloArray<IloArray<IloRangeArray>> constraints_range_positionR;		    	// Constraints (2R) -- Reduces the range of the position depending on the leaf which is selected -- z[t][i][d] <= x[t][i] * rangeUp_{tid}  for all i, t, d
	IloArray<IloRangeArray> constraints_common_position;						// Constraints (3)  -- All positions from all trees should coincide (with zz position) -- zz[d] = sum(i, z[t][i][d]) for all t, d
	IloRange constraint_majority;                                               // Constraint  (4)  -- OtherClass should be in majority

	// Create variables x[t][i]
	x = IloArray<IloNumVarArray>(env, params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		x[t] = IloNumVarArray(env, (int)randomForest->leafRegions[t].size());
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
		{
			int myUBV = 1;
			for (int dFeat : nonTrivialFeaturesBeforeSplits)
			{
				if (randomForest->leafRegions[t][i].range[dFeat].first > topRightCell[dFeat] || randomForest->leafRegions[t][i].range[dFeat].second < bottomLeftCell[dFeat])
				{
					myUBV = 0;
					break;
				}
			}
			//name << "x_" << t << "_" << i;
			x[t][i] = IloNumVar(env, 0, myUBV, IloNumVar::Bool, name.str().c_str());
			name.str(""); // Clean name
		}
	}

	// Create variables z[t][i][d]
	z = IloArray<IloArray<IloNumVarArray>>(env, params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		z[t] = IloArray<IloNumVarArray>(env, (int)randomForest->leafRegions[t].size());
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
		{
			z[t][i] = IloNumVarArray(env, nbNonTrivialFeatures);
			for (int dd = 0; dd < nbNonTrivialFeatures; dd++)
			{
				int dFeat = vectorNonTrivialFeatures[dd];
				//name << "z_" << t << "_" << i << "_" << d;
				z[t][i][dd] = IloNumVar(env, 0, topRightCell[dFeat], IloNumVar::Float, name.str().c_str());
				name.str(""); // Clean name
			}
		}
	}

	// Create variables zz[d]
	zz = IloNumVarArray(env, nbNonTrivialFeatures);
	for (int dd = 0; dd < nbNonTrivialFeatures; dd++)
	{
		int dFeat = vectorNonTrivialFeatures[dd];
		//name << "zz_" << d;
		zz[dd] = IloNumVar(env, bottomLeftCell[dFeat], topRightCell[dFeat], IloNumVar::Float, name.str().c_str());
		name.str(""); // Clean name
	}

	// Constraints (1)
	constraints_selection_leaf = IloRangeArray(env, params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
			expr += x[t][i];
		//name << "selection_leaf_" << t;
		constraints_selection_leaf[t] = IloRange(env, 1, expr, 1, name.str().c_str());
		name.str("");
		expr.clear();
	}
	model.add(constraints_selection_leaf);

	// Constraints (2L-2R)
	constraints_range_positionL = IloArray<IloArray<IloRangeArray>>(env, params->nbTrees);
	constraints_range_positionR = IloArray<IloArray<IloRangeArray>>(env, params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		constraints_range_positionL[t] = IloArray<IloRangeArray>(env, (int)randomForest->leafRegions[t].size());
		constraints_range_positionR[t] = IloArray<IloRangeArray>(env, (int)randomForest->leafRegions[t].size());
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
		{
			constraints_range_positionL[t][i] = IloRangeArray(env, nbNonTrivialFeatures);
			constraints_range_positionR[t][i] = IloRangeArray(env, nbNonTrivialFeatures);
			for (int dd = 0; dd < nbNonTrivialFeatures; dd++)
			{
				int dFeat = vectorNonTrivialFeatures[dd];
				expr = z[t][i][dd] - x[t][i] * randomForest->leafRegions[t][i].range[dFeat].first;
				//name << "constraints_range_positionL_" << t << "_" << i << "_" << d;
				constraints_range_positionL[t][i][dd] = IloRange(env, 0, expr, IloInfinity, name.str().c_str());
				name.str("");
				expr.clear();

				expr = x[t][i] * randomForest->leafRegions[t][i].range[dFeat].second - z[t][i][dd];
				//name << "constraints_range_positionR_" << t << "_" << i << "_" << d;
				constraints_range_positionR[t][i][dd] = IloRange(env, 0, expr, IloInfinity, name.str().c_str());
				name.str("");
				expr.clear();
			}
			model.add(constraints_range_positionL[t][i]);
			model.add(constraints_range_positionR[t][i]);
		}
	}

	// Constraints (3)
	constraints_common_position = IloArray<IloRangeArray>(env, params->nbTrees);
	for (int t = 0; t < params->nbTrees; t++)
	{
		constraints_common_position[t] = IloRangeArray(env, nbNonTrivialFeatures);
		for (int dd = 0; dd < nbNonTrivialFeatures; dd++)
		{
			int dFeat = vectorNonTrivialFeatures[dd];
			expr = zz[dd];
			for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
				expr -= z[t][i][dd];
			//name << "constraints_common_position_" << t << "_" << d;
			constraints_common_position[t][dd] = IloRange(env, 0, expr, 0, name.str().c_str());
			name.str("");
			expr.clear();
		}
		model.add(constraints_common_position[t]);
	}

	// The goal is to obtain a point which has "otherClass" in greater quantity than "pureClass"
	// We set this as our last constraint
    // Constraints (4)
	for (int t = 0; t < params->nbTrees; t++)
	{
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
		{
			if (randomForest->leafRegions[t][i].classification == otherClass)
				expr += x[t][i];
			else if (randomForest->leafRegions[t][i].classification == pureClass)
				expr -= x[t][i];
		}
	}
	//name << "constraint_majority";
	int minRequired = 0;
	if (otherClass > pureClass) minRequired = 1;
	constraint_majority = IloRange(env, minRequired, expr, IloInfinity, name.str().c_str());
	name.str("");
	expr.clear();
	model.add(constraint_majority);

	expr.end(); // Free the memory used by expr
	IloCplex cplex(model); // Create the solver object
	cplex.setOut(env.getNullStream()); // Do not display traces
	cplex.setParam(IloCplex::Param::Threads, 1); // Run on a single thread
	// cplex.exportModel("model-1-Continuous.lp"); // Exports the model to a file (should only be used for debugging)

	// Solve takes value true if Cplex found a feasible solution to the model
	bool mySolution = cplex.solve();

	// Therefore, return true when there is no solution to the model
	return !mySolution;
}


bool MIPCertificate::preFilterMinMax(const std::set<int> & nonTrivialFeaturesBeforeSplits, const std::vector<std::vector<double>> & orderedHyperplanes, const std::vector<unsigned short int> & bottomLeftCell, const std::vector<unsigned short int> & topRightCell, int pureClass, int otherClass)
{
	int minPure = 0 ;
	int maxOther = 0 ;

	// First calculate the MIN number of occurences of class "pureClass" in the RF
	for (int t = 0; t < params->nbTrees; t++)
	{
		bool isPureGuaranteed     = true;
		bool isOtherClassPossible = false;
		for (int i = 0; i < (int)randomForest->leafRegions[t].size(); i++)
		{
			bool isDisjoint = false;
			for (int d : nonTrivialFeaturesBeforeSplits)
			{
				if (randomForest->leafRegions[t][i].range[d].first > topRightCell[d] || randomForest->leafRegions[t][i].range[d].second < bottomLeftCell[d])
				{
					isDisjoint = true;
					break;
				}
			}
			if (!isDisjoint)
			{
				if (randomForest->leafRegions[t][i].classification != pureClass)
					isPureGuaranteed = false;
				if (randomForest->leafRegions[t][i].classification == otherClass)
					isOtherClassPossible = true;
			}
		}

		if (isPureGuaranteed) minPure++;
		if (isOtherClassPossible) maxOther++;
	}

	if (pureClass < otherClass) return (minPure >= maxOther); // If class is of smallest index then equality is enough
	else return (minPure > maxOther); // Otherwise need strict inequality
}
