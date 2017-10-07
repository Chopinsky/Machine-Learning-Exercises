from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
#import pydot
import numpy as np

def train():
    feature = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    return clf.fit(feature, labels)

def test(input):
    clf = train()
    target = [input]
    result = clf.predict(target)

    if result[0] == 1:
        print("Apple")
    else:
        print("Orange")

#test([160, 0])

def iris_train():
    iris = load_iris()

    #header - feature
    #print iris.feature_names
    #header - species
    #print iris.target_names

    #for i in range(len(iris.target)):
    #    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

    test_idx = [0, 75, 100]

    # training data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    #testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    print(test_target)
    print(clf.predict(test_data))

    return clf

def visualization():
    clf = iris_train()
    iris = load_iris()

    dot_data = StringIO()
    tree.export_graphviz(clf,
                            out_file=dot_data,
                            feature_names=iris.feature_names,
                            class_names=iris.target_names,
                            filled=True, rounded=True, impurity=False)

    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("iris.pdf")

visualization()
