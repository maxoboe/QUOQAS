from sklearn import linear_model
import numpy as np
from collections import namedtuple
tokenized_row = namedtuple('tokenized_row', 'sent_count sentences word_count words')
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import csv
from gensim.models.doc2vec import Doc2Vec
Document = namedtuple('ScoreDocument', 'words tags')

def test_batch(test_regressors, test_targets, model, successes, false_pos, false_neg):
    test_predictions = model.predict(test_regressors)
    rounded_predictions = np.rint(test_predictions)
    for i in range(len(rounded_predictions)):
        if rounded_predictions[i] == 1 and test_targets[i] == 0: false_pos += 1
        if rounded_predictions[i] == 0 and test_targets[i] == 1: false_neg += 1
        if rounded_predictions[i] == test_targets[i]: successes += 1
    return successes, false_pos, false_neg

filenames = ['combined_train_test.p', 'r_train_so_test.p', 'so_train_r_test.p',
            'so_alone.p', 'reddit_alone.p']

def load_files(filename):
    with open(filename, 'rb') as pfile:
        train, test = pickle.load(pfile)
    directory_name = filename.split('.p')[0]
    with open(directory_name + "/tokenized_dict.p", 'rb') as pfile:
        train_token_dict, test_token_dict = pickle.load(pfile)
    with open(directory_name + "/body_vectorizer.p", 'rb') as pfile:
        body_vectorizer = pickle.load(pfile) 
    with open(directory_name + "/title_vectorizer.p", 'rb') as pfile:
        title_vectorizer = pickle.load(pfile)   
    return (train, test, train_token_dict, test_token_dict, body_vectorizer, title_vectorizer)


def load_docmodels(filename):
    directory_name = filename.split('.p')[0]
    titles = Doc2Vec.load(directory_name + '/titles.doc2vec')
    bodies = Doc2Vec.load(directory_name + '/bodies.doc2vec')
    title_sentences = Doc2Vec.load(directory_name + '/title_sentences.doc2vec')
    body_sentences = Doc2Vec.load(directory_name + '/body_sentences.doc2vec')
    return titles, bodies, title_sentences, body_sentences


def run_minibatch(dataframe, model, token_dict, body_vectorizer, title_vectorizer, is_train, 
                  incl_words, incl_counts, incl_global_doc = False, incl_local_doc = False, 
                 titles = None, bodies = None, title_sentences = None, body_sentences = None):
    max_batch_size = 1000
    counter = 0
    successes, false_pos, false_neg = 0, 0, 0
    
    feature_length = 0
    
    if incl_words:
        empty_response = title_vectorizer.transform([])
        title_length = empty_response.shape[1]
        empty_response = body_vectorizer.transform([])
        body_length = empty_response.shape[1]
        feature_length += body_length + title_length
    if incl_counts:
        feature_length += 4 # word count, sent count for title and body
    if incl_global_doc:
        feature_length += 200 # Each doc vec has size 100
    if incl_local_doc:
        feature_length += 200 
        
    
    regressors = np.empty([max_batch_size, feature_length])
    targets = np.empty([max_batch_size, ])
    length = len(dataframe.index.values)

    for i in range(length):
        index = dataframe.index.values[i]
        row = token_dict[index]
        title_words = row[0].words
        body_words = row[1].words
        if len(body_words) is 0: body_words = [""]
        title_vectorization = title_vectorizer.transform(title_words).toarray()[0]
        body_vectorization = body_vectorizer.transform(body_words).toarray()[0]
        features = np.array([])
        if incl_words:
            features = np.concatenate((features, title_vectorization, body_vectorization))
        if incl_counts:
            features = np.concatenate((features, [row[0].word_count, row[0].sent_count, 
                                                  row[1].word_count, row[1].sent_count]))
        if incl_global_doc:
            body_vector = bodies.docvecs[index]
            title_vector = titles.docvecs[index]
            features = np.concatenate((features, title_vector, body_vector))
        if incl_local_doc:
            body_vector = np.zeros([100,])
            title_vector = np.zeros([100,])
            for a in range(row[0].sent_count):
                title_vector = title_vector + title_sentences.docvecs[str(index) + '_' +str(a)]
            for a in range(row[1].sent_count):
                body_vector = body_vector + body_sentences.docvecs[str(index) + '_' +str(a)]
            features = np.concatenate((features, title_vector, body_vector))
        regressors[counter] = features
        targets[counter] = dataframe['answer_good'].values[i]
        counter += 1
        if counter == max_batch_size:
            if is_train:
                model.partial_fit(regressors, targets, classes=np.array([0, 1]))
            else:
                successes, false_pos, false_neg = test_batch(regressors, targets, model, 
                                                        successes, false_pos, false_neg)
            if length - i < max_batch_size:
                batch_size = length % max_batch_size
            else:
                batch_size = max_batch_size
            regressors = np.empty([batch_size, feature_length])
            targets = np.empty([batch_size, ])
            counter = 0
    if is_train:
        return model
    else:
        return successes, false_pos, false_neg

def test_and_train(filename, incl_words=False, incl_counts=False, incl_global_doc = False, incl_local_doc = False):
    model = linear_model.PassiveAggressiveClassifier()
    titles, bodies, title_sentences, body_sentences = load_docmodels(filename)
    (train, test, train_token_dict, test_token_dict, body_vectorizer, title_vectorizer) = load_files(filename)
    model = run_minibatch(train, model, train_token_dict, body_vectorizer, title_vectorizer, is_train=True,
                        incl_words=incl_words, incl_counts=incl_counts, 
                        incl_global_doc=incl_global_doc, incl_local_doc=incl_local_doc,
                        titles=titles, bodies=bodies, title_sentences=title_sentences, body_sentences=body_sentences)
    return run_minibatch(test, model, test_token_dict, body_vectorizer, title_vectorizer, is_train=False,
                        incl_words=incl_words, incl_counts=incl_counts, 
                        incl_global_doc=incl_global_doc, incl_local_doc=incl_local_doc,
                        titles=titles, bodies=bodies, title_sentences=title_sentences, body_sentences=body_sentences)

with open('local_doc_alone.csv', 'w+', newline="") as csvfile:
    fieldnames = ['Test Name', 'Success Rate', 'false +', 'false -']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for name in filenames:
        successes, false_pos, false_neg = test_and_train(name, incl_words=False, incl_counts=False,
                                                         incl_global_doc=False, incl_local_doc=True)
        success_rate = float(successes) / (successes + false_pos + false_neg)
        writer.writerow({'Test Name': name, 'Success Rate': success_rate,
                         'false +': false_pos, 'false -': false_neg})