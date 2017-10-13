import dataio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


defaultSeed = 537
splitSeed = 142


def draw_hist(white, red):
    fig, ax = plt.subplots(1, 2)

    ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red wine')
    ax[1].hist(white.alcohol, 10, facecolor='green', ec='black', lw=0.5, alpha=0.5, label='White wine')

    fig.suptitle("Distribution of Alcohol in % Vol")

    ax[0].set_ylim([0, 1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")

    plt.show()


def draw_scatter(white, red):
    np.random.seed(defaultSeed)

    red_labels = np.unique(red['quality'])
    white_labels = np.unique(white['quality'])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    red_colors = np.random.rand(6, 4)
    white_colors = np.append(red_colors, np.random.rand(1, 4), axis=0)

    for i in range(len(red_colors)):
        redy = red['alcohol'][red.quality == red_labels[i]]
        redx = red['volatile acidity'][red.quality == red_labels[i]]
        ax[0].scatter(redx, redy, c=red_colors[i])

    for i in range(len(white_colors)):
        whitey = white['alcohol'][white.quality == white_labels[i]]
        whitex = white['volatile acidity'][white.quality == white_labels[i]]
        ax[1].scatter(whitex, whitey, c=white_colors[i])

    ax[0].set_title("Red Wine")
    ax[0].set_xlim([0, 1.7])
    ax[0].set_ylim([5, 15.5])
    ax[0].set_xlabel("Volatile Acidity")
    ax[0].set_ylabel("Alcohol")

    ax[1].set_title("White Wine")
    ax[1].set_xlim([0, 1.7])
    ax[1].set_ylim([5, 15.5])
    ax[1].set_xlabel("Volatile Acidity")
    ax[1].set_ylabel("Alcohol")
    ax[1].legend(white_labels, loc='best', bbox_to_anchor=(1.3, 1))

    fig.subplots_adjust(top=0.85, wspace=0.7)

    plt.show()


def identify_wine(peek):
    remote_white, remote_red, local_white, local_red = dataio.get_data_sets()

    # load data sets
    white = dataio.load_wine_data(local_white, remote_white)
    red = dataio.load_wine_data(local_red, remote_red)

    # peek data
    if peek is True:
        draw_hist(white, red)
        draw_scatter(white, red)

    # combining data sets
    red['type'] = 1
    white['type'] = 0
    wine_set = red.append(white, ignore_index=True)

    # find correlations and display heatmap
    corr = wine_set.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    #plt.show()
    #print(corr)

    X = wine_set.ix[:, 0:11]
    y = np.ravel(wine_set.type)

    # split the data up into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=splitSeed)

    print("Done!")
