# MIT License

# Copyright(c) 2020 Toni Pacheco and Maximilian Schiffer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sklearn.tree as tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#################################################################################
# importing a txt file
#################################################################################
def importTreeCollection(datasetName, silent=False):
    treeIndex = -1
    new_n_nodes = []
    new_children_left = []
    new_children_right = []
    new_feature = []
    new_threshold = []
    new_node_depth = []
    new_is_leaves = []
    new_nodeValues = []
    new_majorityClass = []
    with open(datasetName, "r") as inputFile:
        data = inputFile.readlines()
        for index in range(len(data)):
            data[index] = data[index][:-1].split()
        for index in range(len(data)):
            
            if index <= 5:
                if (data[index][0] == "DATASET_NAME:"):
                    new_dataset = data[index][1]
                if (data[index][0] == "ENSEMBLE:"):
                    new_ensemble = data[index][1] 
                if (data[index][0] == "NB_TREES:"):
                    new_numOfTrees = int(data[index][1])
                if (data[index][0] == "NB_FEATURES:"):
                    new_numOfFeatures = int(data[index][1])
                if (data[index][0] == "NB_CLASSES:"):
                    new_numOfClasses = int(data[index][1])
                if (data[index][0] == "MAX_TREE_DEPTH:"):
                    new_maxTreeDepth = int(data[index][1])
            
            elif (len(data[index]) == 0):
                if not silent:
                    print("checking for new tree")
    
            elif ((len(data[index]) == 2) and (data[index][0][:5] == "[TREE")):
                if not silent:
                    print("Evaluating new tree")
                # increase the tree index
                treeIndex = treeIndex + 1
                if len(data[index+1]) == 1:
                    data[index+1] = data[index+1][0].split(':')
                new_n_nodes.append(int(data[index+1][1]))
                # add containers for the new tree
                new_children_left.append([])
                new_children_right.append([])
                new_feature.append([])
                new_threshold.append([])
                new_node_depth.append([])
                new_is_leaves.append([])
                new_nodeValues.append([])
                new_majorityClass.append([])
    
    
            elif(len(data[index]) == 8):
                new_children_left[treeIndex].append(int(data[index][2]))
                new_children_right[treeIndex].append(int(data[index][3]))
                new_feature[treeIndex].append(int(data[index][4]))
                new_threshold[treeIndex].append(float(data[index][5]))
                new_node_depth[treeIndex].append(int(data[index][6]))
                new_majorityClass[treeIndex].append(int(data[index][7]))
                if (data[index][1] == 'LN'):
                    new_is_leaves[treeIndex].append(True)
                else:
                    new_is_leaves[treeIndex].append(False)
    return new_dataset, new_ensemble, new_numOfTrees, new_numOfFeatures, new_numOfClasses, new_maxTreeDepth, new_n_nodes, new_children_left, new_children_right, new_feature, new_threshold, new_node_depth, new_is_leaves, new_nodeValues, new_majorityClass
################################################################################



#################################################################################
# creating classifier from file
#################################################################################
def compute_info_from_dataset(X, y, n_features, n_classes, n_outputs, maxTreeDepth,
                              n_nodes, children_left, children_right, features,
                              thresholds, values, compute_score=False):
    
    def compute_samples(cur, X, y, n_samples, impurities):
        
        if children_left[cur] == children_right[cur]:
            n_samples[cur] = len(y)
            if compute_score:
                for i in range(n_classes[0]):
                    values[cur, 0, i] = np.count_nonzero(y==i)
        else:
            if children_left[cur] != -1:
                idxs = np.where(X[:, features[cur]] <= thresholds[cur])[0]
                compute_samples(children_left[cur], X[idxs], y[idxs], n_samples, impurities)
            if children_right[cur] != -1:
                idxs = np.where(X[:, features[cur]] >thresholds[cur])[0]
                compute_samples(children_right[cur], X[idxs], y[idxs], n_samples, impurities)
            
            n_samples[cur] = n_samples[children_left[cur]] + n_samples[children_right[cur]]
            values[cur] = values[children_left[cur]] + values[children_right[cur]]
        
        if len(y) > 0:
            for i in range(n_classes[0]):
                count = np.count_nonzero(y==i)
                p = count/len(y)
                impurities[cur] += p*(1-p)
        else:
            impurities[cur] = 1
            
    
    n_samples = np.zeros_like(features)
    impurities = np.zeros_like(thresholds)
    pred = np.zeros((n_nodes, n_classes[0]), dtype=np.int32)
    compute_samples(0, X, y, n_samples, impurities)
    
    n_samples_norm = np.zeros_like(children_left)
    
    return impurities, n_samples, n_samples_norm


#################################################################################
# creating classifier from file
#################################################################################
def create_nodes(n_nodes, n_outputs, n_classes, children_left,
                 children_right, features, thresholds, actual_values,
                 impurities, n_samples, n_samples_norm, pruning):
    
    def _create_node(cur, out_nodes, out_values, pruning):
        l = children_left[cur]
        r = children_right[cur]
        
        if pruning:
            while (l != r):
                if n_samples[l] == 0:
                    cur = r
                    l = children_left[cur]
                    r = children_right[cur]
                elif n_samples[r] == 0:
                    cur = l
                    l = children_left[cur]
                    r = children_right[cur]
                else:
                    break
        
        if l == r:
            tup = (-1, -1, -1, -1, impurities[cur], n_samples[cur], n_samples_norm[cur])
            if cur != 0:
                out_nodes.append(tup)
                out_values.append(actual_values[cur])
            else:
                out_nodes[0] = tup
                out_values[0] = actual_values[cur]
            return 0, cur
        else:
            dl, cl = _create_node(l, out_nodes, out_values, pruning)
            l = len(out_nodes)-1
            dr, cr = _create_node(r, out_nodes, out_values, pruning)
            r = len(out_nodes)-1
            
            #are there two leaves with same classification?
            actual_values[cur] = actual_values[cl]+actual_values[cr]
            if (pruning and dl==0 and dr==0 and (np.argmax(actual_values[cl]) == np.argmax(actual_values[cr]))):
                dr = dl = -1
                out_nodes.pop()
                out_nodes.pop()
                out_values.pop()
                out_values.pop()
                tup = (-1, -1, -1, -1, impurities[cur], n_samples[cur], n_samples_norm[cur])
            else:
                tup = (l, r, features[cur], thresholds[cur], impurities[cur], n_samples[cur], n_samples_norm[cur])
                

            if cur != 0:
                out_nodes.append(tup)
                out_values.append(actual_values[cur])
            else:
                out_nodes[0] = tup
                out_values[0] = actual_values[cur]
            return max([dl, dr])+1, cur

    
    
    out_nodes = [None]
    out_values = [None]
    depth, _ = _create_node(0, out_nodes, out_values, pruning)
    
    n_nodes = len(out_nodes)
    values = np.zeros((n_nodes, n_outputs, n_classes[0]), dtype=np.float64)
    nodes = np.zeros(n_nodes, dtype=[
                               ('left_child', '<i8'),
                               ('right_child', '<i8'),
                               ('feature', '<i8'),
                               ('threshold', '<f8'),
                               ('impurity', '<f8'),
                               ('n_node_samples', '<i8'),
                               ('weighted_n_node_samples', '<f8')])
    
    for i in range(n_nodes):
        nodes[i] = out_nodes[i]
        values[i,0,:] = out_values[i]
    
    return nodes, values , depth


#################################################################################
# creating classifier from file
#################################################################################
def build_tree(X, y, n_features, n_classes, n_outputs, maxTreeDepth,
              n_nodes, children_left, children_right, features,
               thresholds, values, pruning=False, compute_score=False):
    
    d = {}
    d["nodes"] = np.zeros(n_nodes, dtype=[
                               ('left_child', '<i8'),
                               ('right_child', '<i8'),
                               ('feature', '<i8'),
                               ('threshold', '<f8'),
                               ('impurity', '<f8'),
                               ('n_node_samples', '<i8'),
                               ('weighted_n_node_samples', '<f8')])
    
    # fill values
    actual_values = np.zeros((n_nodes, n_outputs, n_classes[0]), dtype=np.float64)
    for i in range(n_nodes):
        for c in range(n_classes[0]):
            actual_values[i][0][c] = 1 if c==values[i] else 0
            
    # create nodes
    impurities, n_samples, n_samples_norm = compute_info_from_dataset(
            X, y, n_features, n_classes, n_outputs, maxTreeDepth,
            n_nodes, children_left, children_right, features,
            thresholds, actual_values, compute_score
    )
    
    d["nodes"], d["values"], d["max_depth"]  = create_nodes(n_nodes, n_outputs, n_classes,
                                                            children_left, children_right,
                                                            features, thresholds, actual_values,
                                                            impurities, n_samples, n_samples_norm, pruning)
    d["node_count"] = len(d["nodes"])
    
    d["nodes"] = np.array(d["nodes"])

    tree_ = tree._tree.Tree(n_features, n_classes, n_outputs)
    tree_.__setstate__(d)
    return tree_


#################################################################################
# creating classifier from file
#################################################################################
def build_classifier(trees):
    
    def build_decision_tree(t):
        dt = DecisionTreeClassifier(random_state=0)
        dt.n_features_ = t.n_features
        dt.n_outputs_ = t.n_outputs
        dt.n_classes_ = t.n_classes[0]
        dt.classes_ = np.array([x for x in range(dt.n_classes_)])
        dt.tree_ = t
        return dt
    
    if len(trees) > 1:
        clf = RandomForestClassifier(random_state=0, n_estimators=len(trees))
        clf.estimators_ = [build_decision_tree(t) for t in trees]
        clf.n_features_ = trees[0].n_features
        clf.n_outputs_ = trees[0].n_outputs
        clf.n_classes_ = trees[0].n_classes[0]
        clf.classes_ = np.array([x for x in range(clf.n_classes_)])
    else:
        clf = build_decision_tree(trees[0])
    return clf


#################################################################################
# creating classifier from file
#################################################################################
def classifier_from_file(fn, X, y, pruning=False, compute_score=False, num_trees=-1):
    dataset, ensemble, numOfTrees, numOfFeatures, \
    numOfClasses, maxTreeDepth, n_nodes, children_left, \
    children_right, features, thresholds, node_depths, \
    is_leaves, nodeValues, majorityClass = importTreeCollection(fn, silent=True)
    
    n_features = numOfFeatures
    n_classes = [numOfClasses]
    n_outputs = 1
    
    trees = []
    for i in range(numOfTrees):
        t = build_tree(X, y, n_features, np.array(n_classes, dtype=np.intp), n_outputs,
                       maxTreeDepth, n_nodes[i], children_left[i],
                       children_right[i], features[i], thresholds[i],
                       majorityClass[i], pruning, compute_score)
        trees.append(t)
    
    trees = trees if num_trees == -1 else trees[:num_trees]
    return build_classifier(trees)
