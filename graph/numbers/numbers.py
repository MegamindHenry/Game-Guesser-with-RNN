import matplotlib.pyplot as plt

data = {'1': 2183, '2': 3576, '3': 3889, '4': 1767,
        '5': 768, '6': 350, '7': 92, '8': 44,
        '9': 9, '10': 8, '11': 1}
names = list(data.keys())
values = list(data.values())

plt.figure(figsize=(18, 5))

plt.bar(names, values)
plt.savefig('numbers.png')