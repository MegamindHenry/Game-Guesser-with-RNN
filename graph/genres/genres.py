import matplotlib.pyplot as plt

data = {'Indie': 8242, 'Action': 5147, 'Casual': 4302, 'Adventure': 4139,
        'Simulation': 2780, 'Strategy': 2492, 'RPG': 2151, 'Free to Play': 984,
        'Early Access': 956, 'Sports': 673, 'Violent': 607, 'Massively Multiplayer': 467}
names = list(data.keys())
values = list(data.values())

plt.figure(figsize=(18, 5))

plt.bar(names, values)
plt.savefig('genres.png')