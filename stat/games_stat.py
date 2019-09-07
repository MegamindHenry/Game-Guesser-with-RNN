import json
import math
import nltk
import collections

with open('../data/games.json', 'r', encoding='utf8') as fp:

    raw_data = json.load(fp)

print('total app: {}'.format(len(raw_data)))

with open('../processed_data/games_train.json', 'r', encoding='utf8') as fp:
    train_data = json.load(fp)

print('train app: {}'.format(len(train_data)))

with open('../processed_data/games_test.json', 'r', encoding='utf8') as fp:
    test_data = json.load(fp)

print('test app: {}'.format(len(test_data)))

data = train_data + test_data

length = len(data)

print('total app: {}'.format(length))

max_short = 0
min_short = math.inf
total_short = 0

max_detailed = 0
min_detailed = math.inf
total_detailed = 0

genres_freq = {}

genres_num = {}
total_tags = 0



for d in data:
    total_short += len(nltk.tokenize.word_tokenize(d['short_description']))
    total_detailed += len(nltk.tokenize.word_tokenize(d['detailed_description']))

    for g in d['genres']:
        genres_freq[g] = genres_freq.get(g, 0)+1

    genres_length = len(d['genres'])
    genres_num[str(genres_length)] = genres_num.get(str(genres_length), 0)+1

    total_tags += genres_length


genres_freq = collections.OrderedDict(genres_freq)

print('================')

for g in sorted(genres_freq, key=genres_freq.get, reverse=True):
    print(g, genres_freq[g])

print('================')

print('average detailed: {:.2f}'.format(total_detailed/length))
print('average short: {:.2f}'.format(total_short / length))


print('test app: {}'.format(len(data)))

print('==================')

for k, v in genres_num.items():
	print(k, v)

print('average tags: {:.2f}'.format(total_tags/length))