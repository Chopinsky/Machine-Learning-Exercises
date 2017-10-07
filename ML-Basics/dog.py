import numpy as np
import matplotlib.pyplot as plt

grehounds = 500
labs = 500

grey_height = 25 + 4 * np.random.rand(grehounds)
lab_height = 24 + 4 * np.random.rand(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()