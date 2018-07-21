import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def svm_test():
    digits = datasets.load_digits()
    model = svm.SVC(gamma=0.001, C=100)

    x, y = digits.data[:-1], digits.target[:-1]
    model.fit(x, y)

    prediction = model.predict(digits.data[:-1])
    n = -4

    print("Prediction: ", prediction[n])
    plt.imshow(digits.images[n], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def wine_test(use_forest, to_peek):
    wine = datasets.load_wine()

    features = wine.data
    labels = wine.target

    if to_peek:
        peek(features, labels, wine.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    if use_forest:
        model = tree.DecisionTreeClassifier()
    else:
        model = RandomForestClassifier()

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(predictions)
    print(y_test)
    print(model.score(x_test, y_test))


def peek(features, labels, names):
    print("Number of entries: ", len(features))
    for name in names:
        print(name[:10], "  \t")

    print("Class")
    for feature, label in zip(features, labels):
        for f in features:
            print(f, "\t\t")
        print(label)


if __name__ == "__main__":
    # svm_test()

    wine_test(True, False)

    print("\nAll done!")
