import numpy as np

from LSTM import LSTM
from utils.data_converter import *
from utils.preprocess import *
from crf import *

params = {
    "pretrain": True, # change to False to try random embeddings
    "learning_rate": 0.0001,
    "hidden_size": 200, # hidden size of the nn
    "cell": "GRU", # or LSTM, changes the type of nn cell
    "num_layers": 1,
    "model": "BiLSTM", # BiLSTM, LSTM or CRF
    "keep_ratio": 0.5,
    "tags": "BIOES", # can be IOB, BIOS or BIOES
    "crf": True, # change it to False to use typical loss
    "char": True, # use character embeddings
    "char_method": "BiLSTM", # or CNN
    "features": True,
    "char_hidden": 100
}


def main():
    # reads the dataset
    conll_sents = read_conll('data/eng.train')
    conll_tags = get_column(conll_sents, -1)
    conll_words = get_column(conll_sents, 0)

    test_sents = read_conll('data/eng.testb')
    test_tags = get_column(test_sents, -1)

    if params["tags"] == "BIO":
        conll_tags = tags_from_conll(conll_tags, scheme='bio')
        test_tags = tags_from_conll(test_tags, scheme="bio")
    elif params["tags"] == "BIOES":
        conll_tags = tags_from_conll(conll_tags, scheme='bioes')
        test_tags = tags_from_conll(test_tags, scheme='bioes')

    conll_add1 = get_column(conll_sents, 1)
    conll_add2 = get_column(conll_sents, 2)

    if params["model"] == "CRF":
        X = np.array(conll_sents)
        y = np.array(conll_tags)
        #X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=33)

        crf(X, y, test_sents, test_tags)
    else:

        print("Train data includes", len(conll_sents), "sentences")
        unique_tags = set(tag for item in conll_tags for tag in item)
        print(len(unique_tags), "distinct tags found:", unique_tags)
        unique_POS = set(tag for item in conll_add1 for tag in item)
        POS_dict = {pos: idx for idx, pos in enumerate(unique_POS)}

        unique_POS2 = set(tag for item in conll_add2 for tag in item)
        POS_dict2 = {pos: idx for idx, pos in enumerate(unique_POS2)}

        print("Finding distinct words in the data")
        vocab, chars, params["max_char"], params["max_len"] = get_vocabs_chars(conll_words)
        params["max_char"] = 20
        print("There are", len(vocab), "different words in the dataset")

        if params["pretrain"]:
            print("Loading GloVe pretrained vectors")
            embeddings = read_embedding(vocab, 'embeddings/glove.300.txt')
        else:
            embeddings = None

        char_embeddings = read_embedding(chars, 'embeddings/glove.char.txt')


        print("Converting tags to numbers")
        conll_tags = tags_to_id(conll_tags, unique_tags)

        X = np.array(conll_sents)
        y = np.array(conll_tags)
        print("Splitting data to train and test")
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, range(len(conll_sents)),
                                                                                         test_size=0.2, random_state=33)
        X_train = list(X_train)
        X_test = list(X_test)

        if params["features"]:
            print("Extracting features")
            train_feat = list()
            test_feat = list()
            testb_feat_x = list()
            # generates features for each word
            for sent_idx, sent in enumerate(conll_sents):
                sent_feat = [get_features(sent, idx, POS_dict, POS_dict2) for idx in range(len(sent))]
                while len(sent_feat) < params["max_len"]:
                    # default feature vector for padding tokens
                    pad = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 1])
                    sent_feat.append(pad/ sum(pad))
                if sent_idx in indices_train:
                    train_feat.append(np.array(sent_feat))
                else:
                    test_feat.append(np.array(sent_feat))
            max_test = max(len(sent) for sent in test_sents)
            for sent_idx, sent in enumerate(test_sents):
                sent_feat = [get_features(sent, idx, POS_dict, POS_dict2) for idx in range(len(sent))]
                while len(sent_feat) < max_test:
                    # default feature vector for padding tokens
                    pad = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 1])
                    sent_feat.append(pad/ sum(pad))
                testb_feat_x.append(np.array(sent_feat))
            params["feat_len"] = 10
            train_feat = just_batch(np.array(train_feat))
            test_feat = just_batch(np.array(test_feat))
            testb_feat = just_batch(np.array(testb_feat_x))

        else:
            train_feat = list()
            test_feat = list()
            testb_feat = list()

        if params["char"]:
            print("Generating character embeddings")
            X_train_char, train_char_lengths = ids_to_char(chars, vocab, X_train, params["max_char"], params["max_len"])
            X_test_char, test_char_lengths = ids_to_char(chars, vocab, X_test, params["max_char"], params["max_len"])
            testb_char, testb_char_lengths = ids_to_char(chars, vocab, test_sents, params["max_char"], max_test)

            train_char = just_batch(X_train_char)
            len_batch = just_batch(train_char_lengths)
            train_char_batch = list(zip(train_char, len_batch))

            test_char = just_batch(X_test_char)
            test_len_batch = just_batch(test_char_lengths)
            test_char_batch = list(zip(test_char, test_len_batch))

            testb_char_b = just_batch(testb_char)
            testb_len_batch = just_batch(testb_char_lengths)
            testb_char_batch = list(zip(testb_char_b, testb_len_batch))
        else:
            train_char_batch = list()
            test_char_batch = list()
            testb_char_batch = list()

        X_train = words_to_id(vocab, X_train)
        X_test = words_to_id(vocab, X_test)
        testb = words_to_id(vocab, test_sents)

        train_batch = get_batches(X_train, y_train, max_length=params["max_len"], pad_idx=vocab.index("<pad>"))
        test_batch = get_batches(X_test, y_test, max_length=params["max_len"], pad_idx=vocab.index("<pad>"))
        testb_batch = get_batches(testb, max_length=max_test, pad_idx=vocab.index("<pad>"))

        lstm = LSTM(params, vocab, embeddings, len(unique_tags), chars, char_embeddings)
        lstm.build()
        conll_sents = read_conll('data/eng.train')
        lstm.run_model(train_batch, test_batch, train_char_batch, test_char_batch,
                       [conll_sents[i] for i in indices_test], {idx: tag for idx, tag in enumerate(unique_tags)}, train_feat, test_feat,
                       testb_feat, testb_char_batch, testb_batch, read_conll('data/eng.testb'), max_test)


if __name__ == '__main__':
  main()