import fasttext

model = fasttext.train_supervised('../processed_data/games_fasttext.train')

precision = []
recall = []
f1 = []

def print_results(N, p, r):
    # print("N\t" + str(N))
    print("P\t{:.4f}".format(p))
    print("R\t{:.4f}".format(r))
    print('F\t{:.4f}'.format(2*p*r/(p+r)))
    print('==================')


def results(N, p, r):
    precision.append('{:.4f}'.format(p))
    recall.append('{:.4f}'.format(r))
    f1.append('{:.4f}'.format(2*p*r/(p+r)))
#
# print('\n\n9 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=9))
# print_results(*model.test('../processed_data/games_fasttext.test', k=9))
#
# print('\n\n8 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=8))
# print_results(*model.test('../processed_data/games_fasttext.test', k=8))
#
# print('\n\n7 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=7))
# print_results(*model.test('../processed_data/games_fasttext.test', k=7))
#
# print('\n\n6 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=6))
# print_results(*model.test('../processed_data/games_fasttext.test', k=6))
#
# print('\n\n5 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=5))
# print_results(*model.test('../processed_data/games_fasttext.test', k=5))
#
# print('\n\n4 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=4))
# print_results(*model.test('../processed_data/games_fasttext.test', k=4))
#
# print('\n\n3 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=3))
# print_results(*model.test('../processed_data/games_fasttext.test', k=3))
#
# print('\n\n2 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=2))
# print_results(*model.test('../processed_data/games_fasttext.test', k=2))
#
# print('\n\n1 labels')
# # print_results(*model.test('../processed_data/games_fasttext.train', k=1))
# print_results(*model.test('../processed_data/games_fasttext.test', k=1))


for i in range(1, 10):
    results(*model.test('../processed_data/games_fasttext.test', k=i))

print('fasttext_precision = [{}]'.format(', '.join(precision)))
print('fasttext_recall = [{}]'.format(', '.join(recall)))
print('fasttext_f1 = [{}]'.format(', '.join(f1)))

