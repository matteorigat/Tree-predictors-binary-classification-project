import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    def __init__(self, max_depth=None, min_samples_split=2, feature_names=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = feature_names

    def fit(self, X, y):
        # Ensure X, y are converted to numpy array
        X = np.array(X)
        y = np.array(y)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        print("Depth grow tree:", depth)
        num_samples, num_features = X.shape
        if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or (np.unique(y).size == 1):
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        feat_idxs = np.random.choice(num_features, num_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

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
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        num_samples = len(y)
        num_samples_left, num_samples_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (num_samples_left / num_samples) * e_left + (num_samples_right / num_samples) * e_right
        ig = parent_entropy - child_entropy
        return ig

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

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
    tree = DecisionTree(max_depth=5, min_samples_split=2, feature_names=X_train.columns)

    # Fit the model
    tree.fit(X_train.values, y_train.values)

    # Make predictions
    y_pred = tree.predict(X_test.values)
    print("\n\n")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\n\n")

    # Print the tree
    tree.print_tree()

    # Visualize the tree
    dot = Digraph()
    dot = tree.visualize_tree(dot)
    dot.render('mushroom_tree', format='png', view=True)




