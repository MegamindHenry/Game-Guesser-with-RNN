import matplotlib.pyplot as plt
precision = [0.4289, 0.5700, 0.6648, 0.7379, 0.8036, 0.8650, 0.9413, 0.9819, 1.0000]
recall = [0.8660, 0.7467, 0.6412, 0.5395, 0.4332, 0.3024, 0.1496, 0.0779, 0.0496]
f1 = [0.5737, 0.6465, 0.6528, 0.6233, 0.5630, 0.4481, 0.2582, 0.1443, 0.0946]
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.plot(x, precision, label='precision')
plt.plot(x, recall, label='recall')
plt.plot(x, f1, label='f1 score')

plt.yticks(y)

plt.ylabel('Score')
plt.xlabel('Threshold')
plt.legend(loc='right')
plt.savefig('threshold.png')