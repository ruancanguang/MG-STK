import numpy as np
import random as rd
a = np.linspace(0, 1.0, num=1000)
a = a.tolist()
print(a)
rd.shuffle(a)
print(a)
# print(a.mean())
