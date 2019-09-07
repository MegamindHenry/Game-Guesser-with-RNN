import matplotlib.pyplot as plt
precision = [0.7706, 0.7048, 0.6068, 0.5185, 0.4464, 0.3940, 0.3507, 0.3166, 0.2877]
recall = [0.2804, 0.5130, 0.6625, 0.7548, 0.8124, 0.8603, 0.8934, 0.9217, 0.9422]
f1 = [0.4112, 0.5938, 0.6334, 0.6148, 0.5762, 0.5404, 0.5037, 0.4713, 0.4407]
fasttext_precision = [0.6823, 0.5560, 0.4805, 0.4424, 0.3893, 0.3585, 0.3319, 0.2991, 0.2739]
fasttext_recall = [0.2483, 0.4047, 0.5246, 0.6440, 0.7084, 0.7828, 0.8455, 0.8709, 0.8973]
fasttext_f1 = [0.3641, 0.4684, 0.5016, 0.5244, 0.5024, 0.4918, 0.4767, 0.4453, 0.4197]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.plot(x, precision, label='precision', color='red')
plt.plot(x, recall, label='recall', color='blue')
plt.plot(x, f1, label='f1 score', color='green')

plt.plot(x, fasttext_precision, label='fasttext precision', color='red', linestyle='--')
plt.plot(x, fasttext_recall, label='fasttext recall', color='blue', linestyle='--')
plt.plot(x, fasttext_f1, label='fasttext f1 score', color='green', linestyle='--')

plt.yticks(y)

plt.ylabel('Score')
plt.xlabel('K')
plt.legend(loc='right')
plt.savefig('k.png')