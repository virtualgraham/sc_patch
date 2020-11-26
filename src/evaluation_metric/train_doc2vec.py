import gensim

from csv import reader


def read_corpus(fname, tokens_only=False):
    with open(fname, 'r') as read_obj:
        csv_reader = reader(read_obj)
        next(csv_reader)
        for i, s in enumerate(csv_reader):
            tokens = s[0].split(' ')
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
   
train_corpus = list(read_corpus('sequences.csv'))
# print(train_corpus[:2])

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)