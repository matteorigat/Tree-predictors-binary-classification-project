# Tree Predictors for Binary Classification

This project involves the implementation of tree predictors from scratch for binary classification to determine whether mushrooms are poisonous using the Mushroom dataset. The tree predictors use single-feature binary tests as the decision criteria at any internal node.

## Project Structure

### Node Class

The node class should possess the following attributes and procedures:
- **Constructor**: Initializes the node (empty or with given attributes).
- **Left and Right Children**: Nodes that act as children.
- **Leaf Flag**: A flag to check if the node is a leaf.
- **Decision Criterion/Test**: A function taking a data point (e.g., a numpy vector) as input and returning a Boolean value as output.

### Tree Predictor Class

The tree predictor class should contain the following attributes and procedures:
- **Constructor**: Initializes the tree predictor (possibly passing information on which decision criteria/tests can be adopted on each feature).
- **Splitting Criterion**: Selects both the leaf to expand and the decision criterion to adopt in the new internal node (e.g., Gini index, scaled entropy, etc.).
- **Stopping Criterion**: Halts the construction of the decision tree. Examples include maximum tree depth, maximum number of nodes/leaves, a constraint on the weight of leaves (e.g., number of samples reaching the leaf, entropy/impurity of the leaf), minimum entropy/impurity decrease.
- **Training Procedure**: Trains the tree predictor on a given training set.
- **Evaluation Procedure**: Evaluates the tree predictor on a given validation/test set.

## Implementation Guidelines

1. **Node Class**:
   - Implement a basic class/structure for the nodes.
   - Initialize left and right children nodes.
   - Include a flag to check if the node is a leaf.
   - Implement the decision criterion/test as a function.

2. **Tree Predictor Class**:
   - Implement a constructor for the tree predictor.
   - Implement a splitting criterion for selecting leaves and decision criteria.
   - Implement stopping criteria to halt tree construction.
   - Implement procedures for training and evaluating the tree predictor.
   - Add extra attributes/procedures if necessary.

3. **Training and Evaluation**:
   - Train tree predictors adopting at least 3 reasonable criteria for the expansion of the leaves.
   - Implement at least 2 reasonable stopping criteria.
   - Compute the training error of each tree predictor according to the 0-1 loss.

4. **Hyperparameter Tuning**:
   - Perform hyperparameter tuning according to the splitting and stopping criteria adopted.
   - Use a sound procedure for hyperparameter tuning for at least one tree predictor.

## Report

Write a report discussing your findings with a focus on:
- Adopted methodology.
- Thorough discussion about the models’ performance.
- Comments on the presence of over/underfitting.
- Techniques to tackle overfitting, such as pruning the tree predictors or appropriate stopping criteria.
