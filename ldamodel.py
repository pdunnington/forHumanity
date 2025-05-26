# import nltk
# nltk.download('punkt')

import pandas as pd
import numpy as np
import json
import glob
import spacy

#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim

import pandas as pd
import numpy as np
import json
import glob
import spacy

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim

# Read the CSV file
incidents = pd.read_csv("/Users/patrickdunnington/Desktop/DS_Capstone/msds_capstone_2024/personal_notebooks/patrick_nb/incidents_clean.csv")

# Convert date column to datetime format
incidents['date'] = pd.to_datetime(incidents['date'])

# Clean dataset
for i in range(1, 565):
    incidents.loc[i, 'reportnumber'] = incidents.loc[i, 'reports'].count(",") + 1

# Clean description column
incidents['clean_description'] = incidents['description'].str.replace('ai', '').str.replace('AI', '')

def lemmatization(incidents, allowed_postags=["NOUN","ADJ","VERB","ADV"]):
    nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
    descript_out = []
    for incident in incidents:
        doc = nlp(incident)
        new_descript = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_descript.append(token.lemma_)
        final = ' '.join(new_descript)
        descript_out.append(final)      
    return (descript_out) 

lemmatized_decript = lemmatization(incidents['clean_description'])

def gen_words(incidents):
    final = []
    for incident in incidents:
        new = gensim.utils.simple_preprocess(incident, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_decript)

id2word = corpora.Dictionary(data_words)

corpus = []

for incident in data_words:
    new = id2word.doc2bow(incident)
    corpus.append(new)

word = id2word[[0][:1][0]]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto')

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
vis