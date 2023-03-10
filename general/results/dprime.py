"""
used to calculate mean d_prime (MDP)
for all trials in 5x2 training split
"""

import os
import statistics as stats
from os.path import expanduser

import matplotlib.pyplot as plt

home = expanduser("~")

x = [28.69, 28.98, 29.28, 30.71, 28.16, 26.30, 28.53, 25.93, 25.51, 24.07]
a = [1.05, 9.92, 1.42, 4.88, 14.36, 23.52, 18.49, 22.45, 8.83, 1.04]
b = [22.38, 23.77, 2.39, 12.8, 20.85, 7.1, 3.79, 1.94, 18.98, 16.67]
c = [17.72, 10.26, 10.46, 19.38, 23.58, 16.61, 4.64, 17.73, 16.46, 15.95]
d = [7.48, 21.91, 14.22, 18.74, 3.62, 18.02, 1.56, 18.46, 8.24, 6.34]

names = ["LO0", "LO1", "LO2", "LO3", "LO4"]
means = [f"mean: {stats.mean(x):.2f}" for x in [x, a, b, c, d]]
names = ["\n".join([x, y]) for x, y in zip(names, means)]

fig, ax = plt.subplots()
# ax.violinplot([a, b, c, d], showextrema=True, showmeans=True)
bplot = ax.boxplot([x, a, b, c, d], patch_artist=True)

colors = ["grey", "pink", "lightblue", "lightgreen", "mediumpurple"]
for box, color in zip(bplot["boxes"], colors):
    box.set_facecolor(color)

plt.xticks([1, 2, 3, 4, 5], names)
ax.set(
    xlabel="Leave Out",
    ylabel="d_prime",
    title="Distribution of d_prime in 5x2 training split",
)

fig.tight_layout()
plt.savefig(os.path.join(home, "figures", "MDP.png"))
