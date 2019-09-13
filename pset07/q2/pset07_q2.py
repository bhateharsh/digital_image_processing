import numpy as np
import matplotlib.pyplot as plt

x = [i for i in range(8)]
orgHist = [8, 20, 17, 4, 3, 6, 4, 2]
eqHist = [0, 8, 0, 20, 0, 17+4, 3+6, 4+2]

plt.figure()
plt.subplot(1,2,1)
plt.bar(x,orgHist,align='center')
plt.title('Original Histogram', fontsize='x-small')
plt.ylabel('Occurence')
plt.xlabel('Pixel Value')
plt.subplot(1,2,2)
plt.bar(x,eqHist,align='center')
plt.title('Equalized Histogram', fontsize='x-small')
plt.ylabel('Occurence')
plt.xlabel('Pixel Value')
plt.show()