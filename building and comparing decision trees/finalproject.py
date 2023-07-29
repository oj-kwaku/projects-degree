import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import math
import psutil
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("haberman.csv")
#df = pd.read_csv("wine.csv")
# Convert the data to a NumPy array
data = df.values

# Split the data into features (X) and labels (y)
X = data[:, :-1]
y = data[:, -1]
# X = data.iloc[:, :-1].values
# Y = data.iloc[:, -1]. values.reshape(-1,1)


class DecisiontreeNode():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, content=None):
        ''' constructor '''

        # for decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.content = content

# creating the decision tree class. This will take a fit method, predict method
# To achieve this however, we would need to recursively build our tree, calculate our split and how to do it.
# We would need methods to do all these, and methods to calculate our gini index and entropy too
# Although the DecisionTreeClassifier from Sckit learn has three criteria for splitting, I would consider two.

class DecisionTree_Classifier():
    '''Creating the decision tree class'''
    def __init__(self, min_samples_split=None, max_depth=None, criterion = None):
        ''' constructor '''

        self.min_samples_split = min_samples_split if min_samples_split is not None else 2
        self.max_depth = max_depth if max_depth is not None else 2
        self.criterion = criterion

        # initialize the root of the tree
        self.root = None

    def fit(self, X, y):
        ''' function to train the tree '''

        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        '''the grow_tree method to build our tree '''
        n_labels = len(set(y))
        # setting samples_total to number of rows and features_total to number of columns of X
        samples_total, features_total = np.shape(X)

        # setting conditions for stopping
        # depth can't be more than maximum depth and number of samples cannot be less than the minimum sample split.
        if samples_total < self.min_samples_split or depth >= self.max_depth or n_labels == 1:
            leaf_value = self._cal_leaf(y)
            # return leaf node
            return DecisiontreeNode(content=leaf_value)

            # find the best split
        best_split = self.get_best_split(X, y, features_total)
        # check if information gain is positive
        if best_split["_info_gain"] > 0:
            # grow left recursively
            L_subtree = self.grow_tree(best_split["X_left"], best_split["y_left"], depth + 1)
            # grow right recursively
            R_subtree = self.grow_tree(best_split["X_right"], best_split["y_right"], depth + 1)
            # returning the decision node
            return DecisiontreeNode(best_split["feature_index"], best_split["threshold"],
                                    L_subtree, R_subtree, best_split["_info_gain"])


    def _cal_leaf(self, y):
        '''function to calculate leaf node content'''
        # finding the most common class in y
        return Counter(y).most_common(1)[0][0]


    def get_best_split(self, X, y, num_features):
        ''' function to find the best split '''

        # initialize best split as an empty dictionary
        best_split = {"feature_index": None, "threshold":None, "X_left":None, "y_left": None,"X_right":None,
                      "y_right": None,"_info_gain": -math.inf}
        # initializing the maximum info gain to the least possible value
        max_info_gain = -float("inf")

        # iterate over all the features
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            # get unique values of current feature
            possible_thresholds = np.unique(feature_values)
            # iterating over the unique values
            for threshold in possible_thresholds:
                # get current split
                X_left, y_left, X_right, y_right = self.data_split(X, y, feature_index, threshold)

                # checking to see if child is not empty
                if len(X_left) > 0 and len(X_right) > 0:
                    # calculate the information gained
                    _info_gain = self.information_gained(y, y_left, y_right)

                # updating split

                    if _info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["X_left"] = X_left
                        best_split["y_left"] = y_left
                        best_split["X_right"] = X_right
                        best_split["y_right"] = y_right
                        best_split["_info_gain"] = _info_gain
                        max_info_gain = _info_gain


        # return best split
        return best_split

    def data_split(self, X, y, feature_index, threshold):
        ''' function to split the data '''
        # creating boolean array based on features and threshold
        left = np.where(X[:, feature_index] <= threshold)
        right = np.where(X[:, feature_index] > threshold)

        X_left, y_left = X[left], y[left]
        X_right, y_right = X[right], y[right]

        return X_left, y_left, X_right, y_right

    def information_gained(self, y, y_left, y_right):
        ''' function to calculate the information gain based on criterion selected '''
        info_gain = None
        if self.criterion == "gini":
            prob_left = len(y_left) / len(y)
            prob_right = len(y_right) / len(y)

            left_gini = self.gini_index(y_left)
            right_gini = self.gini_index(y_right)

            info_gain = self.gini_index(y) - prob_left * left_gini - prob_right * right_gini
        elif self.criterion == "entropy":
            entprob_left = len(y_left) / len(y)
            entprob_right = len(y_right) / len(y)

            entropy_left =self.entropy(y_left)
            entropy_right = self.entropy(y_right)

            info_gain = self.entropy(y) - entprob_left * entropy_left - entprob_right * entropy_right
        else:
            print("select either entropy or gini as criterion")


        return info_gain

    def gini_index(self, y):
        ''' function to calculate the gini impurity '''
        unique_labels = np.unique(y)
        gini = 1
        for label in unique_labels:
            prob = len(np.where(y == label)[0]) / len(y)
            gini -= prob ** 2

        return gini



    def entropy(self, y):
        class_probabilities = [len(y[y == cls]) / len(y) for cls in np.unique(y)]
        return -sum([p * np.log2(p) for p in class_probabilities if p > 0])




    def predict(self, X):
        ''' function to predict y '''
        # Setting an empty list to collect all predictions
        predictions = []
        # iterating and traversing through tree
        for x in X:
            predictions.append(self.prediction_helper(x, self.root))
        return predictions


    def prediction_helper(self, x, tree):
        """helper function to traverse through tree to make predictions"""
        while tree.content is None:
            feature_val = x[tree.feature]
            if feature_val <= tree.threshold:
                tree = tree.left
            else:
                tree = tree.right
        return tree.content






X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.1, random_state=41)

###############
#Fit the model
classifier = DecisionTree_Classifier(min_samples_split=2, max_depth=4, criterion="gini")
classifier.fit(X_train, Y_train)
#classifier.print_tree()

################
#test model
Y_pred = classifier.predict(X_test)
print(Y_pred)
print(accuracy_score(Y_test, Y_pred))
print (precision_score(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))


'''For the implemented Decisiontree_classifier'''
# # Lists to store the results
problem_size = []
training_times = []
memory_usages = []
accuracies = []
precisions = []
recalls = []
F1_score = []


'''iterating problem size'''
test_size = np.arange(0.1,0.9,0.1)
for size in test_size:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=size, random_state=41)
    problem_size.append(size)
    cls = DecisionTree_Classifier(min_samples_split=2, max_depth=3, criterion="gini")
    # Measure the training time
    start_time = time.time()
    cls.fit(X_train,y_train)
    end_time = time.time()
    training_time = end_time - start_time
    training_times.append(training_time)

    #using psutil to calculate memory used
    def fit_classifier(cls, X_train, y_train):
        # Train the classifier
        cls.fit(X_train,y_train)

    # Get the current run
    process = psutil.Process()

    # Train the classifier
    fit_classifier(cls, X_train, y_train)

    # Get the memory used running an iter
    memory_used = process.memory_info().rss

    memory_usages.append(memory_used)

    #make predictions on test
    y_pred = cls.predict(X_test)
    # Calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    #Store the results
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    F1_score.append(f1)

#checking if the length is the same as in the Sk-learn decision tree
# print(len(memory_usages))
# print(len(training_times))
# print(len(accuracies))
# print(len(recalls))

# Create a dataframe with the results
# results1s = {"memory used(bytes)": memory_usages,
#            "training time": training_times,
#            "accuracy": accuracies,
#            "precision": precisions,
#            "recall": recalls,
#            "F1": F1_score,
#            "size" : problem_size
#             }
#
# dataset1s = pd.DataFrame(results1s)
#
# # exporting file
# dataset1s.to_csv("Implementation_size")



'''Iterating over criteria'''
train_times = []
mem_usages = []
accuracies_c = []
precisions_c = []
recalls_c =[]
F1_score_c = []
criteria_c = []
'''changing criteria'''
criteria = ["gini", "entropy"]
for criterion in criteria:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    cls = DecisionTree_Classifier(min_samples_split=2, max_depth=3, criterion=criterion)
    criteria_c.append(criterion)


    # Measure the training time
    start_time = time.time()
    cls.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    train_times.append(training_time)


    # using psutil to calculate memory used
    def fit_classifier(cls, X_train, y_train):
        # Train the classifier
        cls.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    fit_classifier(cls, X_train, y_train)

    # Get the memory used running an iter
    memory_used = process.memory_info().rss

    mem_usages.append(memory_used)

    # make predictions on test
    y_pred = cls.predict(X_test)
    # Calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Store the results
    accuracies_c.append(accuracy)
    precisions_c.append(precision)
    recalls_c.append(recall)
    F1_score_c.append(f1)

#checking if the length is the same as in the Sk-learn decision tree
# print(len(mem_usages))
# print(len(train_times))
# print(len(accuracies_c))
# print(len(recalls_c))

# Create a dataframe with the results
# results1c = {"memory used(bytes)": mem_usages,
#            "training time": train_times,
#            "accuracy": accuracies_c,
#            "precision": precisions_c,
#            "recall": recalls_c,
#            "F1": F1_score_c,
#            "criterion" : criteria_c}
#
# dataset1c = pd.DataFrame(results1c)
#
# # exporting file
# dataset1c.to_csv("implementation_criterion")

'''iterating on min_samples_split'''

train_times_m = []
mem_usages_m = []
accuracies_m = []
precisions_m = []
recalls_m = []
F1_score_m = []
sample_split = []


min_split = np.arange(2, 50, 1)

for split in min_split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    cls = DecisionTree_Classifier(min_samples_split=split, max_depth=3, criterion="gini")
    sample_split.append(split)

    # Measure the training time
    start_time = time.time()
    cls.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    train_times_m.append(training_time)


    # using psutil to calculate memory used
    def fit_classifier(cls, X_train, y_train):
        # Train the classifier
        cls.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    fit_classifier(cls, X_train, y_train)

    # Get the memory used running an iter
    memory_used = process.memory_info().rss

    mem_usages_m.append(memory_used)

    # make predictions on test
    y_pred = cls.predict(X_test)
    # Calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Store the results
    accuracies_m.append(accuracy)
    precisions_m.append(precision)
    recalls_m.append(recall)
    F1_score_m.append(f1)

#checking if the length is the same as in the Sk-learn decision tree
# print(len(mem_usages_m))
# print(len(train_times_m))
# print(len(accuracies_m))
# print(len(recalls_m))
#
# # Create a dataframe with the results
# results1m = {"memory used(bytes)": mem_usages_m,
#            "training time": train_times_m,
#            "accuracy": accuracies_m,
#            "precision": precisions_m,
#            "recall": recalls_m,
#            "F1": F1_score_m,
#            "min_sample_split" : sample_split}
#
# dataset1m = pd.DataFrame(results1m)
#
# # exporting file
# dataset1m.to_csv("implementation_minSamp")
#


"""iterating on max_depth"""
train_times_d = []
mem_usages_d = []
accuracies_d = []
precisions_d = []
recalls_d = []
F1_score_d = []
maximum_depth = []

maxi_depth = np.arange(3,25,1)
for depth in maxi_depth:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    cls = DecisionTree_Classifier(min_samples_split=2, max_depth=depth, criterion= "gini")
    maximum_depth.append(depth)

    # Measure the training time
    start_time = time.time()
    cls.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    train_times_d.append(training_time)


    # using psutil to calculate memory used
    def fit_classifier(cls, X_train, y_train):
        # Train the classifier
        cls.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    fit_classifier(cls, X_train, y_train)

    # Get the memory used running an iter
    memory_used = process.memory_info().rss

    mem_usages_d.append(memory_used)

    # make predictions on test
    y_pred = cls.predict(X_test)
    # Calculate the performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Store the results
    accuracies_d.append(accuracy)
    precisions_d.append(precision)
    recalls_d.append(recall)
    F1_score_d.append(f1)
#checking if the length is the same as in the Sk-learn decision tree
# print(len(mem_usages_d))
# print(len(train_times_d))
# print(len(accuracies_d))
# print(len(recalls_d))


# Create a dataframe with the results
# results1d = {"memory used(bytes)": mem_usages_d,
#            "training time": train_times_d,
#            "accuracy": accuracies_d,
#            "precision": precisions_d,
#            "recall": recalls_d,
#            "F1": F1_score_d,
#            "max_depth" : maximum_depth}
#
# dataset1d = pd.DataFrame(results1d)
#
# # exporting file
# dataset1d.to_csv("implementation_maxdepth")




'''USING SCIKIT LEARN DECISION TREE CLASSIFIER'''
from sklearn.tree import DecisionTreeClassifier
sk_training_times = []
sk_memory_usages = []
sk_accuracies = []
sk_precisions = []
sk_recalls = []
sk_F1_score = []
sk_problem_size = []

'''Iterating on problem size'''
test_size1 = np.arange(0.1,0.9,0.1)
for size in test_size1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=41)
    sk_problem_size.append(size)
    # Set the hyperparameters of the classifier
    decisionTree = DecisionTreeClassifier(min_samples_split=2, max_depth=3, criterion="gini")
    # Measure the training time
    start_time = time.time()
    decisionTree.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    sk_training_times.append(training_time)


    # Measure memory usage

    def sk_fit_classifier(decisionTree, X_train, y_train):
        # Train the classifier
        decisionTree.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    sk_fit_classifier(decisionTree, X_train, y_train)

    # Get the memory used running an iter
    sk_memory_used = process.memory_info().rss

    sk_memory_usages.append(sk_memory_used)

    # Make predictions on the test set
    y_predict = decisionTree.predict(X_test)

    # Calculate the performance metrics
    sk_accuracy = accuracy_score(y_test, y_predict)
    sk_precision = precision_score(y_test, y_predict)
    sk_recall = recall_score(y_test, y_predict)
    sk_F1 = f1_score(y_test, y_predict)
    # Store the results
    sk_accuracies.append(sk_accuracy)
    sk_precisions.append(sk_precision)
    sk_recalls.append(sk_recall)
    sk_F1_score.append(sk_F1)
# checking length to see if it is the same as from the implemented decision tree
# print(len(sk_accuracies))
# print(len(sk_precisions))
# print(len(sk_F1_score))
# print(len(sk_memory_usages))
# print(len(sk_training_times))
# print(len(sk_recalls))
# print(len(sk_problem_size))

#creating a dataframe to export
# sk_results = {"memory used(bytes)": sk_memory_usages,
#            "training time": sk_training_times,
#            "accuracy": sk_accuracies,
#            "precision": sk_precisions,
#            "recall": sk_recalls,
#            "F1": sk_F1_score,
#            "size" : sk_problem_size}
#
# sk_dataset = pd.DataFrame(sk_results)
#
# #Export file as csv
# sk_dataset.to_csv("sk_problemSize")


'''iterating on criteria'''
#list to store results
sk_training_times_c = []
sk_memory_usages_c = []
sk_accuracies_c = []
sk_precisions_c = []
sk_recalls_c = []
sk_F1_score_c = []
sk_criteria = []

criteria1 = ["gini", "entropy"]
for criterion in criteria1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    # Set the hyperparameters of the classifier
    decisionTree = DecisionTreeClassifier(min_samples_split=2, max_depth=3, criterion=criterion)
    sk_criteria.append(criterion)
    # Measure the training time
    start_time = time.time()
    decisionTree.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    sk_training_times_c.append(training_time)


    # Measure memory usage

    def sk_fit_classifier(decisionTree, X_train, y_train):
        # Train the classifier
        decisionTree.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    sk_fit_classifier(decisionTree, X_train, y_train)

    # Get the memory used running an iter
    sk_memory_used = process.memory_info().rss

    sk_memory_usages_c.append(sk_memory_used)

    # Make predictions on the test set
    y_predict = decisionTree.predict(X_test)

    # Calculate the performance metrics
    sk_accuracy = accuracy_score(y_test, y_predict)
    sk_precision = precision_score(y_test, y_predict)
    sk_recall = recall_score(y_test, y_predict)
    sk_F1 = f1_score(y_test, y_predict)
    # Store the results
    sk_accuracies_c.append(sk_accuracy)
    sk_precisions_c.append(sk_precision)
    sk_recalls_c.append(sk_recall)
    sk_F1_score_c.append(sk_F1)

# checking length to see if it is the same as from the implemented decision tree
# print(len(sk_memory_usages_c))
# print(len(sk_training_times_c))
# print(len(sk_accuracies_c))
# print(len(sk_precisions_c))

#creating a dataframe to export
# sk_results = {"memory used(bytes)": sk_memory_usages_c,
#            "training time": sk_training_times_c,
#            "accuracy": sk_accuracies_c,
#            "precision": sk_precisions_c,
#            "recall": sk_recalls_c,
#            "F1": sk_F1_score_c,
#            "criterion" : sk_criteria}
#
# sk_dataset = pd.DataFrame(sk_results)
#
# # Export file as csv
# sk_dataset.to_csv("sk_criteria")


'''Iterating on min_sample_split'''
sk_training_times_s = []
sk_memory_usages_s = []
sk_accuracies_s = []
sk_precisions_s = []
sk_recalls_s = []
sk_F1_score_s = []
sk_minSample_split = []

#range to split over
min_split1 = np.arange(2, 50, 1)
for split in min_split1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    # Set the hyperparameters of the classifier
    decisionTree = DecisionTreeClassifier(min_samples_split=split, max_depth=3, criterion="gini")
    sk_minSample_split.append(split)
    # Measure the training time
    start_time = time.time()
    decisionTree.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    sk_training_times_s.append(training_time)


    # Measure memory usage

    def sk_fit_classifier(decisionTree, X_train, y_train):
        # Train the classifier
        decisionTree.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    sk_fit_classifier(decisionTree, X_train, y_train)

    # Get the memory used running an iter
    sk_memory_used = process.memory_info().rss

    sk_memory_usages_s.append(sk_memory_used)

    # Make predictions on the test set
    y_predict = decisionTree.predict(X_test)

    # Calculate the performance metrics
    sk_accuracy = accuracy_score(y_test, y_predict)
    sk_precision = precision_score(y_test, y_predict)
    sk_recall = recall_score(y_test, y_predict)
    sk_F1 = f1_score(y_test, y_predict)
    # Store the results
    sk_accuracies_s.append(sk_accuracy)
    sk_precisions_s.append(sk_precision)
    sk_recalls_s.append(sk_recall)
    sk_F1_score_s.append(sk_F1)

# #checking length to see if it is the same as from the implemented decision tree
# print(len(sk_memory_usages_s))
# print(len(sk_training_times_s))
# print(len(sk_accuracies_s))
# print(len(sk_precisions_s))

# #creating a dataframe to export
# sk_results = {"memory used(bytes)": sk_memory_usages_s,
#            "training time": sk_training_times_s,
#            "accuracy": sk_accuracies_s,
#            "precision": sk_precisions_s,
#            "recall": sk_recalls_s,
#            "F1": sk_F1_score_s,
#            "min_sample_split" : sk_minSample_split}
#
# sk_dataset = pd.DataFrame(sk_results)
# #
# # Export file as csv
# sk_dataset.to_csv("sk_minSample")


'''Iterating over max_depth'''
sk_training_times_d = []
sk_memory_usages_d = []
sk_accuracies_d = []
sk_precisions_d = []
sk_recalls_d = []
sk_F1_score_d = []
sk_maximum_depth = []

maxi_depth1 = np.arange(3,25,1)
for depth in maxi_depth1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    # Set the hyperparameters of the classifier
    decisionTree = DecisionTreeClassifier(min_samples_split=2, max_depth=depth, criterion="gini")
    sk_maximum_depth.append(depth)
    # Measure the training time
    start_time = time.time()
    decisionTree.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    sk_training_times_d.append(training_time)


    # Measure memory usage

    def sk_fit_classifier(decisionTree, X_train, y_train):
        # Train the classifier
        decisionTree.fit(X_train, y_train)


    # Get the current run
    process = psutil.Process()

    # Train the classifier
    sk_fit_classifier(decisionTree, X_train, y_train)

    # Get the memory used running an iter
    sk_memory_used = process.memory_info().rss

    sk_memory_usages_d.append(sk_memory_used)

    # Make predictions on the test set
    y_predict = decisionTree.predict(X_test)

    # Calculate the performance metrics
    sk_accuracy = accuracy_score(y_test, y_predict)
    sk_precision = precision_score(y_test, y_predict)
    sk_recall = recall_score(y_test, y_predict)
    sk_F1 = f1_score(y_test, y_predict)
    # Store the results
    sk_accuracies_d.append(sk_accuracy)
    sk_precisions_d.append(sk_precision)
    sk_recalls_d.append(sk_recall)
    sk_F1_score_d.append(sk_F1)

# checking length to see if it is the same as from the implemented decision tree
# print(len(sk_memory_usages_d))
# print(len(sk_training_times_d))
# print(len(sk_accuracies_d))
# print(len(sk_precisions_d))


#creating a dataframe to export
# sk_results = {"memory used(bytes)": sk_memory_usages_d,
#            "training time": sk_training_times_d,
#            "accuracy": sk_accuracies_d,
#            "precision": sk_precisions_d,
#            "recall": sk_recalls_d,
#            "F1": sk_F1_score_d,
#            "max_depth" : sk_maximum_depth}
#
# sk_dataset = pd.DataFrame(sk_results)
#
# #Export file as csv
# sk_dataset.to_csv("sk_maxdepth")
