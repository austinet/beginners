
import matplotlib.pyplot as plt

x1 = 5
y1 = [1, 4, 6, 8, 9]

a1, = plt.plot(range(1, x1+1), y1, label='1')
a2, = plt.plot([2, 4, 7, 8, 11], [1, 2, 3, 4, 5], label='2')
plt.ylabel('y')
plt.xlabel('x')

plt.legend(handles=[a1, a2])
plt.show()