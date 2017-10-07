import dataio
import matplotlib.pyplot as plt
import numpy as np


defaultSeed = 537


def drawHist(white, red):
    fig, ax = plt.subplots(1, 2)

    ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red wine')
    ax[1].hist(white.alcohol, 10, facecolor='green', ec='black', lw=0.5, alpha=0.5, label='White wine')

    #fig.subplots_adjust(left=0, right=1, bottom=0, top=5, hspace=0.05, wspace=1)
    fig.suptitle("Distribution of Alcohol in % Vol")

    ax[0].set_ylim([0, 1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")

    plt.show()


def drawScatter(white, red):
    np.random.seed(defaultSeed)

    redLabels = np.unique(red['quality'])
    whiteLabels = np.unique(white['quality'])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    redColors = np.random.rand(6, 4)
    whiteColors = np.append(redColors, np.random.rand(1, 4), axis=0)

    for i in range(len(redColors)):
        redy = red['alcohol'][red.quality == redLabels[i]]
        redx = red['volatile acidity'][red.quality == redLabels[i]]
        ax[0].scatter(redx, redy, c=redColors[i])

    for i in range(len(whiteColors)):
        whitey = white['alcohol'][white.quality == whiteLabels[i]]
        whitex = white['volatile acidity'][white.quality == whiteLabels[i]]
        ax[1].scatter(whitex, whitey, c=whiteColors[i])

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
    ax[1].legend(whiteLabels, loc='best', bbox_to_anchor=(1.3, 1))

    fig.subplots_adjust(top=0.85, wspace=0.7)

    plt.show()


def identifyWine():
    remote_white, remote_red, local_white, local_red = dataio.getDataSets()

    white = dataio.loadWineData(local_white, remote_white)
    red = dataio.loadWineData(local_red, remote_red)

    #drawHist(white, red)
    drawScatter(white, red)

    print("Done!")
