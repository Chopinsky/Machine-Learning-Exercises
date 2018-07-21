import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def iris(peek):
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    if peek:
        print(iris_X[:2, :])
        print(iris_y)

    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

    # Cross validations:
    k_range = range(2,30)
    k_scores = []
    k_losses = []
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(knn, iris_X, iris_y, cv=10, scoring="accuracy")  # classification
        loss = -cross_val_score(knn, iris_X, iris_y, cv=10, scoring="neg_mean_squared_error")  # regression

        k_scores.append(scores.mean())
        k_losses.append(loss.mean())

        # print("Neighbor size: ", i, " -- ", scores.mean())

    # Score plot
    plt.plot(k_range, k_scores)
    plt.show()

    # Loss plot
    plt.plot(k_range, k_losses)
    plt.show()

    # Fitting with 15 neighbors and predict
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    result = knn.predict(X_test)

    with open("model/iris.pickle", "wb") as f:
        pickle.dump(knn, f)

    if peek:
        print(result)
        print(y_test)


def boston(peek):
    boston = datasets.load_boston()
    boston_X = boston.data
    boston_y = boston.target

    boston_X = preprocessing.scale(boston_X)
    boston_y = preprocessing.scale(boston_y)

    X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    result = model.predict(X_test)

    '''
    plt.scatter(y_train, y_train)
    plt.show()
    
    diff = np.abs(y_test - result)
    diff_sum = np.sqrt(np.sum(diff * diff)) / y_test.size
    print(diff_sum)
    '''

    print(model.score(X_test, y_test))

    with open("model/boston.pickle", "wb") as f:
        pickle.dump(model, f)

    if peek:
        print(result[:10])
        print(y_test[:10])


if __name__ == "__main__":
    print("\nIris - KNeighbors: ")
    iris(False)

    print("\nBoston house price - Linear regression:")
    boston(False)

    with open("model/iris.pickle", "rb") as f:
        model = pickle.load(f)

    print("\ndone...")
