import matplotlib.pyplot as plt
fasttext_f1 = [0.6197, 0.6387, 0.6258, 0.6287, 0.6288]
fasttext_recall = [0.6457, 0.6293, 0.6059, 0.6081, 0.6069]
fasttext_precision = [0.5958, 0.6484, 0.6470, 0.6508, 0.6523]

keras_f1 = [0.5645, 0.5446, 0.5533, 0.5686, 0.5799]
keras_recall = [0.5785, 0.5192, 0.5392, 0.5395, 0.5577]
keras_precision = [0.5511, 0.5726, 0.5681, 0.6009, 0.6038]

x = [1, 2, 3, 4, 5]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
x_labels = [32, 64, 128, 256, 512]

plt.plot(x, fasttext_precision, label='fasttext precision', color='red')
plt.plot(x, fasttext_recall, label='fasttext recall', color='blue')
plt.plot(x, fasttext_f1, label='fasttext f1 score', color='green')

plt.plot(x, keras_precision, label='keras precision', color='red', linestyle='--')
plt.plot(x, keras_recall, label='keras recall', color='blue', linestyle='--')
plt.plot(x, keras_f1, label='keras f1 score', color='green', linestyle='--')

plt.xticks(x, x_labels)
plt.yticks(y)

plt.ylabel('Score')
plt.xlabel('Embedding Dimension')
plt.legend(loc='lower right')
plt.savefig('embedding.png')