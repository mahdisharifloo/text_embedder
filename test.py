# !pip install -qU sentence-transformers
# !pip install -qU wikipedia-api
# !pip install -qU hazm
# !pip install -qU clean-text[gpl]
# !pip install -qU emoji

# from preprocessing import cleaning
from IPython import display
# from preprocessing import cleaning

import numpy as np
import pandas as pd

import hazm
import requests
import time

import torch
from sentence_transformers import models, SentenceTransformer, util
from sklearn.cluster import KMeans

def rtl_print(outputs, font_size="15px", n_to_br=False):
    outputs = outputs if isinstance(outputs, list) else [outputs] 
    if n_to_br:
        outputs = [output.replace('\n', '<br/>') for output in outputs]
        
    outputs = [f'<p style="text-align: right; direction: rtl; margin-right: 10px; font-size: {font_size};">{output}</p>' for output in outputs]
    display.display(display.HTML(' '.join(outputs)))

    
def load_st_model(model_name_or_path):
    word_embedding_model = models.Transformer(model_name_or_path)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)
    
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

# Corpus with example sentences
corpus = [
    'مردی در حال خوردن خوراک است.',
    'مردی در حال خوردن یک تکه نان است.',
    "مردی در حال خوردن پاستا است.",
    'دختری بچه ای را حمل می کند.',
    'بچه ای توسط زنی حمل می شود.',
    'زنی در حال نواختن پیانو است.',
    'یک مرد سوار بر اسب است.',
    'مردی در حال سواری بر اسب سفید در مزرعه است.',
    'میمونی در حال نواختن طبل است.',
    "کسی با لباس گوریل مشغول نواختن مجموعه ای از طبل است.",
    'یوزپلنگ به دنبال شکار خود در حال دویدن است.',
    "یوزبلنگ شکار خود را در یک مزرعه تعقیب می کند.",
]

num_clusters = 5

# Load the Sentence-Transformer
embedder = load_st_model('m3hrdadfi/bert-fa-base-uncased-wikinli-mean-tokens')
corpus_embeddings = embedder.encode(corpus, show_progress_bar=True)


# Perform kmean clustering
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, sentences in enumerate(clustered_sentences):
    print(f'Cluster: {i + 1}', '20px')
    print(sentences)
    print('- - ' * 50)