import json
import pickle
from bs4 import BeautifulSoup
import random


# convert json format into fasttext format
def json2fasttext(data, is_short=True):
    lines = []
    for d in data:
        parts = []
        gs = d['genres']
        for g in gs:
            parts.append('__label__{}'.format(g))
        if is_short:
            parts.append(d['short_description'])
        else:
            parts.append(d['detailed_description'])

        lines.append(' '.join(parts))

    return '\n'.join(lines)


# load games
with open('../data/games.json', 'r', encoding='utf8') as fp:
    games = json.load(fp)

print('total games: {}'.format(len(games)))


# read full game info and create genres only info
genres = set()
games_processed = []
descriptions = []

for k, v in games.items():
    if 'genres' not in v['data']:
        continue

    if 'short_description' not in v['data'] or v['data']['short_description'] == '':
        continue

    if 'detailed_description' not in v['data'] or v['data']['detailed_description'] == '':
        continue

    game_genres = []

    for g in v['data']['genres']:
        genres.add(g['description'])
        game_genres.append(g['description'])

    soup = BeautifulSoup(v['data']['detailed_description'], features="html.parser")

    game = {'short_description': v['data']['short_description'],
            'detailed_description': soup.get_text(),
            'genres': game_genres}

    descriptions.append(v['data']['short_description'])
    descriptions.append(soup.get_text())

    games_processed.append(game)

genres = sorted(genres)

print(genres)


# save genres into a file
with open("genres", "wb") as fp:
    pickle.dump(genres, fp)

# save description into file
with open('corpus.txt', 'w+', encoding='utf8') as fp:
    fp.write('\n'.join(descriptions))


# random shuffle and split into test and train
print('total games processed: {}'.format(len(games_processed)))
random.shuffle(games_processed)

games_test = games_processed[:len(games_processed)//5]
games_train = games_processed[len(games_processed)//5:]

print('number of train data: {}'.format(len(games_train)))
print('number of train data: {}'.format(len(games_test)))


# save data into files
games_train_output = json.dumps(games_train, indent=4)
with open('games_train.json', 'w+', encoding='utf8') as fp:
    fp.write(games_train_output)

games_test_output = json.dumps(games_test, indent=4)
with open('games_test.json', 'w+', encoding='utf8') as fp:
    fp.write(games_test_output)

# save fasttext format
games_faxttext_train_output = json2fasttext(games_train)
with open('games_fasttext.train', 'w+', encoding='utf8') as fp:
    fp.write(games_faxttext_train_output)

games_faxttext_test_output = json2fasttext(games_test)
with open('games_fasttext.test', 'w+', encoding='utf8') as fp:
    fp.write(games_faxttext_test_output)