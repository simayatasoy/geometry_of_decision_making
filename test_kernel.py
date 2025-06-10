import numpy as np
from random import randrange
import math
import matplotlib.pyplot as plt
import os
import numpy as np


alpha = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
i = np.random.randint(0,1000)
alpha_i = alpha[i]

v_0 = 1
v_1 = 0.75
v_2 = 0.5
angle_diffs = np.abs((alpha - alpha_i + np.pi) % (2 * np.pi) - np.pi)

kernel_0 = np.cos(np.pi * ((angle_diffs / np.pi) ** v_0))
kernel_0[i] = 0
kernel_1 = np.cos(np.pi * ((angle_diffs / np.pi) ** v_1))
kernel_1[i] = 0
kernel_2 = np.cos(np.pi * ((angle_diffs / np.pi) ** v_2))
kernel_2[i] = 0

plt.figure()

plt.plot(alpha,kernel_0, 'r')
plt.plot(alpha,kernel_1, 'b')
plt.plot(alpha,kernel_2, 'g')
plt.show()
