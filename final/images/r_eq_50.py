import matplotlib
import numpy as np

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Linux Libertine O'
matplotlib.rcParams['mathtext.it'] = 'Linux Libertine O:italic'
matplotlib.rcParams['mathtext.bf'] = 'Linux Libertine Os:bold'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import matplotlib.pyplot as plt

plt.plot(*np.loadtxt("run_tag_accuracy_50.csv",delimiter=",", skiprows=1, usecols=(1,2), unpack=True), linewidth=2.0)
plt.savefig('foo.png')
