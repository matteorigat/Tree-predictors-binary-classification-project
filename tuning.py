from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from graphviz import Digraph


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Value the node has if it is a leaf

    def is_leaf(self):
        return self.value is not None

    def __str__(self, level=0):
        ret = "\t" * level + f"Feature {self.feature}: "
        if self.is_leaf():
            ret += f"Class {self.value}\n"
        else:
            ret += f"Threshold {self.threshold}\n"
            ret += self.left.__str__(level + 1)
            ret += self.right.__str__(level + 1)
        return ret


class DecisionTree:
    def __init__(self, max_depth=None, max_leaf_nodes=None, split_function=None, min_samples_split=2, feature_names=None):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.split_function = split_function
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = feature_names
        self.leaf_count = 0

    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'max_leaf_nodes': self.max_leaf_nodes,
            'split_function': self.split_function,
            'min_samples_split': self.min_samples_split,
            'feature_names': self.feature_names
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        if (self.max_depth is not None and depth >= self.max_depth) or (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes) or (num_samples < self.min_samples_split) or (np.unique(y).size == 1):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feat_idxs = np.random.choice(num_features, num_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        self.leaf_count += 1
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return TreeNode(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._gain(y, X_column, threshold, criterion=self.split_function)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gain(self, y, X_column, split_thresh, criterion):
        criterion_func = {
            'scaled_entropy': self._scaled_entropy,
            'gini': self._gini_impurity,
            'squared': self._squared_impurity,
            'misclassification': self._misclassification
        }.get(criterion)

        parent_criterion = criterion_func(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        y_left, y_right = y[left_idxs], y[right_idxs]
        child_criterion = self._weighted_criterion(y_left, y_right, criterion_func)
        gain = parent_criterion - child_criterion
        return gain

    def _weighted_criterion(self, y_left, y_right, criterion_func):
        n = len(y_left) + len(y_right)
        p_left = len(y_left) / n
        p_right = len(y_right) / n
        return p_left * criterion_func(y_left) + p_right * criterion_func(y_right)

    # From class lectures scaled_ent = - (p/2)*np.log2(p) - ((1-p)/2)*np.log2(1-p) for binary classification
    def _scaled_entropy(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        scaled_ent = -np.sum([(p / 2) * np.log2(p) for p in probs if p > 0])
        return scaled_ent

    #From class lectures it should be 2p(1-p) for binary classification, but let use the general formula for non-binary case
    def _gini_impurity(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        gini = 1.0 - np.sum(probs ** 2)  # gini = np.sum(probs * (1 - probs))
        return gini

    def _misclassification(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        mce = 1.0 - np.max(probs)
        return mce

    #for binary classification    sqrt(p*(1-p))
    def _squared_impurity(self, y):
        hist = np.bincount(y)
        probs = hist / len(y)
        epsilon = 1e-10  # Small constant to avoid multiplying by zero
        sqr = np.sum(np.sqrt((probs + epsilon) * (1 - probs + epsilon)))
        return sqr


    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _print_tree(self, node, depth=0):
        if node is None:
            return
        indent = '  ' * depth
        if node.is_leaf():
            print(indent + f"Leaf: Class {node.value}")
        else:
            if self.feature_names is not None:
                feature_name = self.feature_names[node.feature]
            else:
                feature_name = f"Feature {node.feature}"
            print(indent + f"{feature_name} <= {node.threshold}")
            self._print_tree(node.left, depth + 1)
            print(indent + f"{feature_name} > {node.threshold}")
            self._print_tree(node.right, depth + 1)

    def print_tree(self):
        self._print_tree(self.root)

    def visualize_tree(self, dot=None):
        if dot is None:
            dot = Digraph()

        # Recursive function to add nodes and edges
        def add_nodes_edges(dot, node):
            if node is None:
                return
            if node.is_leaf():
                dot.node(str(id(node)), f"Class {node.value}", shape='ellipse')
            else:
                if self.feature_names is not None:
                    feature_name = self.feature_names[node.feature]
                else:
                    feature_name = f"Feature {node.feature}"
                dot.node(str(id(node)), f"{feature_name} <= {node.threshold}", shape='box')
                if node.left is not None:
                    add_nodes_edges(dot, node.left)
                    dot.edge(str(id(node)), str(id(node.left)), '<=')
                if node.right is not None:
                    add_nodes_edges(dot, node.right)
                    dot.edge(str(id(node)), str(id(node.right)), '>')

        add_nodes_edges(dot, self.root)
        return dot



def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)



if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv('MushroomDataset/secondary_data.csv', delimiter=';')

    # Display the first few rows to verify the data
    # print(data.head())

    # Convert categorical features to one-hot encoding
    data_encoded = pd.get_dummies(data)


    # Separate features and target
    X = data_encoded.drop(['class_p', 'class_e'], axis=1)  # Assuming 'class_p' is the target
    y = data_encoded['class_p']  #.apply(lambda x: 1 if x == 'p' else 0)  # Convert 'p' to 1, 'e' to 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)

    # Instantiate the decision tree classifier
    #tree = DecisionTree(max_depth=5, min_samples_split=2, feature_names=X_train.columns)










    param_grid = [
        {
            'max_depth': [5, 10, 20, 50, 100, 200, 500, 1000],
            'split_function': ['scaled_entropy', 'gini', 'squared']
        },
        {
            'max_leaf_nodes': [5, 50, 100, 200, 500, 1000, 2000, 5000],
            'split_function': ['scaled_entropy', 'gini', 'squared']
        }
    ]

    custom_scorer = make_scorer(zero_one_loss, greater_is_better=False)

    # Crea un oggetto GridSearchCV
    grid_search = GridSearchCV(estimator=DecisionTree(max_depth=None, max_leaf_nodes=None, split_function=None, min_samples_split=2, feature_names=X_train.columns), param_grid=param_grid, cv=3, scoring=custom_scorer, verbose=3)

    # Esegui il tuning sugli iperparametri usando i dati di addestramento
    grid_search.fit(X_train.values, y_train.values)

    # Ottieni i risultati completi della grid search
    results = grid_search.cv_results_
    print("Grid Search Results:")
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"Mean Accuracy: {mean_score:.4f} with params: {params}")

    # Ottieni i migliori iperparametri trovati
    best_params = grid_search.best_params_
    print("\nBest Hyperparameters:", best_params)

    #Addestra un nuovo modello con i migliori iperparametri trovati
    best_tree = DecisionTree(
        max_depth=best_params.get('max_depth'),
        max_leaf_nodes=best_params.get('max_leaf_nodes'),
        split_function=best_params['split_function'],
        min_samples_split=2,  # This is fixed as per the original configuration
        feature_names=X_train.columns
    )

    """
    
    # Addestra un nuovo modello con i migliori iperparametri trovati
    best_tree = DecisionTree(
        max_depth=None,
        max_leaf_nodes=5000,
        split_function="scaled_entropy",
        min_samples_split=2,  # This is fixed as per the original configuration
        feature_names=X_train.columns
    )
    """

    best_tree.fit(X_train.values, y_train.values)

    # Make predictions on the training data
    y_pred = best_tree.predict(X_train.values)

    # Evaluate the model on the training data
    accuracy = accuracy_score(y_train, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")



    # Predict on the testing data
    y_test_pred = best_tree.predict(X_test.values)

    # Evaluate the model on the training data
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Compute the zero-one loss
    train_error = zero_one_loss(y_test.values, y_test_pred)

    print(f"zero one loss on test set with best params: {train_error:.4f}")

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test.values, y_test_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()




    print("\n\n")

    # Print the tree
    #tree.print_tree()

    # Visualize the tree
    dot = Digraph()
    dot = best_tree.visualize_tree(dot)
    dot.render('png/mushroom_tree', format='png', view=True)

