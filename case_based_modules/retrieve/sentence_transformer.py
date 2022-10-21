# From https://github.com/arosh/BM25Transformer/blob/master/bm25.py

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import numpy as np
import scipy.sparse as sp
import spacy
from sklearn.metrics.pairwise import cosine_distances

from sentence_transformers import SentenceTransformer
import os
import csv
import pickle
import time
import faiss
import numpy as np

class SentenceEmbedding:
    def fit(self, corpus, question_train, ids, question_train_ids, model_name):
        self.corpus = corpus
        self.ids = ids
        self.question_ids = question_train_ids
        self.joined_corpus = []
        self.question_train = question_train
        for fact in corpus:
            self.joined_corpus.append(" ".join(fact))

        self.model = SentenceTransformer(model_name)
        #create index for facts embeddings
        self.facts_index = self.build_faiss_index("facts", self.model, self.joined_corpus)
        #create index for cases embeddings
        self.cases_index = self.build_faiss_index("cases", self.model, self.question_train)

    def query(self, query):
        top_k_hits = len(self.ids)
        query_embedding = self.model.encode(query[0])
        #FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = self.facts_index.search(query_embedding, top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        res = []
        for hit in hits[0:top_k_hits]:
            res.append({"id":self.ids[hit['corpus_id']], "score": hit['score']})

        return res

    def question_similarity(self, query):
        top_k_hits = len(self.question_ids)
        query_embedding = self.model.encode(query[0])
        #FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = self.cases_index.search(query_embedding, top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        res = []
        for hit in hits[0:top_k_hits]:
            res.append({"id":self.question_ids[hit['corpus_id']], "score": hit['score']})

        return res


    def build_faiss_index(self, index_type, model, corpus_sentences):
        max_corpus_size = 10000
        embedding_size = 1024    #Size of embeddings
        embedding_cache_path = 'embeddings-size-'+str(max_corpus_size)+'-'+index_type+'.pkl'

        #Defining our FAISS index
        #Number of clusters used for faiss.
        n_clusters =  1

        #We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

        #Number of clusters to explorer at search time.
        index.nprobe = 1

        #Check if embedding cache path exists
        if not os.path.exists(embedding_cache_path):
            # Check if the dataset exists. If not, download and extract
            print("Encode the corpus. This might take a while")
            corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
            print("Store file on disc")
            with open(embedding_cache_path, "wb") as fOut:
                pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
                #'ids': corpus_ids_original,
        else:
            print("Load pre-computed embeddings from disc")
            with open(embedding_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                #corpus_ids_original = cache_data['ids']
                #corpus_sentences = cache_data['sentences']
                corpus_embeddings = cache_data['embeddings']

        ### Create the FAISS index
        print("Start creating FAISS index")
        # First, we need to normalize vectors to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)
        #faiss.write_index(index, filename)

        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)

        print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

        return index
