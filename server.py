import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import gzip
from nltk.corpus import stopwords
import itertools
from nltk import pos_tag
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from nltk import pos_tag
import scipy.sparse as sp
import nltk
from flask_cors import CORS
import json
import nltk

# import gensim
# import pickle
# from gensim.models import Word2Vec
# from gensim import corpora, models, similarities
# from pprint import pprint
# from collections import defaultdict
# from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity, Similarity


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    if request.method == 'POST':
        try:
            query = str(request.form['question'])
            asin_id = str(request.form['asin'])
            #print(query)
        except Exception as e:
            raise e


    qa = pd.read_csv('questions.csv', index_col=0)


    set_of_question = qa[['question','questionType','asin','answer']]


    pd.options.mode.chained_assignment = None
    set_of_question.loc[set_of_question['questionType'] == 'open-ended', 'questionType'] = 0
    set_of_question.loc[set_of_question['questionType'] == 'yes/no', 'questionType'] = 1
    # print(set_of_question)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(set_of_question['question'], set_of_question['questionType'], random_state = 0)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression


    vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    len(vect.get_feature_names())


    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    feature_names = np.array(vect.get_feature_names())
    sorted_coef_index = model.coef_[0].argsort()


    str_question = query

    question_type = model.predict(vect.transform([str_question]))

    ##### For asin specific search
    toy_corpus = set_of_question[(set_of_question['asin'] == asin_id) & (set_of_question['questionType'] == question_type[0])]
    toy_corpus.reset_index(inplace=True)
    toy_corpus = toy_corpus['answer']

    #query_docs = qa['question']
    query_docs = [query]


    def build_feature_matrix(documents, feature_type='frequency',
                             ngram_range=(1, 1), min_df=0.0, max_df=1.0):
        feature_type = feature_type.lower().strip()
        if feature_type == 'binary':
            vectorizer = CountVectorizer(binary=True, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'frequency':
            vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                         max_df=max_df, ngram_range=ngram_range)
        elif feature_type == 'tfidf':
            vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                         ngram_range=ngram_range,stop_words='english')
        else:
            raise Exception("Wrong feature type entered. Possible values:'binary', 'frequency','tfidf'")
        feature_matrix = vectorizer.fit_transform(documents).astype(float)
        return vectorizer, feature_matrix



    def compute_corpus_term_idfs(corpus_features, norm_corpus):
        dfs = np.diff(sp.csc_matrix(corpus_features, copy=True).indptr)
        dfs = 1 + dfs # to smoothen idf later
        total_docs = 1 + len(norm_corpus)
        idfs = 1.0 + np.log(float(total_docs) / dfs)
        return idfs


    vectorizer, corpus_features = build_feature_matrix(toy_corpus,
                                                       feature_type='tfidf')

    query_docs_features = vectorizer.transform(query_docs)

    doc_lengths = [len(doc.split()) for doc in toy_corpus]
    avg_dl = np.average(doc_lengths)

    corpus_term_idfs = compute_corpus_term_idfs(corpus_features,
                                                toy_corpus)

    def compute_bm25_similarity(doc_features, corpus_features,
                                corpus_doc_lengths, avg_doc_length,
                                term_idfs, k1=1.5, b=0.75, top_n=3):
        # get corpus bag of words features
        corpus_features = corpus_features.toarray()
        # convert query document features to binary features
        # this is to keep a note of which terms exist per document
        doc_features = doc_features.toarray()[0]
        doc_features[doc_features >= 1] = 1
        # compute the document idf scores for present terms
        doc_idfs = doc_features * term_idfs
        # compute numerator expression in BM25 equation
        numerator_coeff = corpus_features * (k1 + 1)    
        numerator = np.multiply(doc_idfs, numerator_coeff)
        # compute denominator expression in BM25 equation
        denominator_coeff =  k1 * (1 - b +
                                    (b * (corpus_doc_lengths /
                                            avg_doc_length)))
        denominator_coeff = np.vstack(denominator_coeff)
        denominator = corpus_features + denominator_coeff
        # compute the BM25 score combining the above equations
        bm25_scores = np.sum(np.divide(numerator,
                                       denominator),
                             axis=1)
        # get top n relevant docs with highest BM25 score
        top_docs = bm25_scores.argsort()[::-1][:top_n]
        top_docs_with_score = [(index, round(bm25_scores[index], 3))
                                for index in top_docs]
        return top_docs_with_score


    print('Document Similarity Analysis using BM25')
    print('='*60)
    for index, doc in enumerate(query_docs):
        doc_features = query_docs_features[index]
        # print(doc_features)
        top_similar_docs = compute_bm25_similarity(doc_features,corpus_features,doc_lengths,avg_dl,corpus_term_idfs,k1=1.5, b=0.75,top_n=2)
        print(top_similar_docs)
    print('Question:', doc)
    #print('Top', len(top_similar_docs), 'similar docs:')
    print('Answer')
    print('-'*40)
    final_predictions = []
    for doc_index, sim_score in top_similar_docs:
        print(doc_index, sim_score)
        final_predictions.append((str(doc_index), str(sim_score), toy_corpus[doc_index]))
        print('Doc num: {} BM25 Score: {}\nDoc: {}'.format(doc_index+1,sim_score, toy_corpus[doc_index]))
        print('-'*40)
        break
    reponses = jsonify(final_predictions)




    # sentiment = defaultdict(int)
    # nlp = set(stopwords.words('english'))
    # # stoplist = STOP_WORDS
    # dictionary = corpora.Dictionary.load('./data/reviews_corpus.dict')
    # corpus = corpora.MmCorpus('./data/reviews_corpus.mm')
    # corpus_2 = corpora.SvmLightCorpus('./data/reviews_corpus.svmlight')
    # index = Similarity.load('./data/reviews_corpus.index')
    # tfidf = models.TfidfModel.load('./data/reviews_corpus_tfidf_2.tfidf_model')
    # lsi = models.LsiModel.load('./data/reviews_corpus_lsi_2.model')




    # reviews_df = pd.read_pickle('./data/reviews_sentiment.pickle')

    # class MyCorpus(object):
    #     def __iter__(self):
    #         for line in open('./data/reviews.txt'):
    #             # assume there's one document per line, tokens separated by whitespace
    #             yield dictionary.doc2bow(line.lower().split())

    # def tokenize(query): 
    #     return [word for word in query.lower().split() if word not in nlp]


    # # query = 'NEED YOUR INPUT NIKHIL'
    # asin = asin_id

    # def review_output(query, asin):
    #     query_bow = dictionary.doc2bow(tokenize(query))
    #     query_tfidf = tfidf[query_bow]
    #     query_lsi = lsi[query_tfidf]
        
    #     index.num_best = 3
    #     sim_ls = (index[query_lsi])
        
    #     ls = []
    #     for i in sim_ls:
    #         ls.append(i[0])

    #     asin_ls = []
    #     for x in list(test.query('asin == '+'\"' +asin +'\"').index):
    #         asin_ls.append(x)
        
    #     output = set(ls) & set(asin_ls)

    #     if len(output)>0:
    #         for m in output:
    #             mout = pd.DataFrame(reviews_df.loc[m]).T
    #             mout_other = pd.DataFrame(reviews_df.query('asin == '+'\"' +asin +'\"').sort_values(by='helpful', ascending=False)[:2])
    #             new_mout = pd.concat([mout, mout_other], axis=0)
    #             new_mout = new_mout.loc[:,['reviewText', 'net_sentiment']]
    #             return new_mout
    #     else:
    #         print('There are no relevant reviews for the selected answer. However the following reviews may be helpful \n')
    #         other_revs = pd.DataFrame(reviews_df.query('asin == '+'\"' +asin +'\"').sort_values(by='helpful', ascending=False)[:3])
    #         other_revs = other_revs.loc[:,['reviewText', 'net_sentiment']]
    #         return other_revs

    # reviews_final = review_output(query,asin)
    # responses.append((reviews_final))
    return(reponses)
