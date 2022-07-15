import pandas as pd
import spacy
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class DataProvider:
    def __init__(self, data, positive_threshold=3):
        self._data = data
        self._threshold = positive_threshold

    def get_positive_review(self, product_id='all'):
        if product_id == 'all':
            data = self._data.loc[self._data.loc[:, 'star_rating'] >= self._threshold]
        else:
            data = self._data.loc[self._data.loc[:, 'product_id'] == product_id]
            data = data.loc[data.loc[:, 'star_rating'] >= self._threshold]

        return data

    def get_negative_review(self, product_id='all'):
        if product_id == 'all':
            data = self._data.loc[self._data.loc[:, 'star_rating'] <= self._threshold]
        else:
            data = self._data.loc[self._data.loc[:, 'product_id'] == product_id]
            data = data.loc[data.loc[:, 'star_rating'] < self._threshold]

        return data

    def get_data(self, product_id):
        return self._data.loc[self._data.loc[:, 'product_id'] == product_id]

class PreProcessor:
    def __init__(self):
        pass

    def lower_text(self, data, column):
        data.loc[:, column] = data.loc[:, column].str.lower()
        return data


class Processor:
    def __init__(self):
        pass

    def extract_tokens_plus_meta(self, doc: spacy.tokens.doc.Doc):
        """Extract tokens and metadata from individual spaCy doc."""

        text = [
            i.lemma_ for i in doc if not any([i.is_punct, i.is_bracket, i.is_stop])
        ]

        return " ".join(text)

    # add custom stop words


class Modeller:
    def __init__(self, load_small=True):
        
        if load_small:
            try:
                self.nlp = spacy.load("en_core_web_sm")
        except: 
            spacy.cli.download("en_core_web_sm")
            
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = spacy.load("en_core_web_lg")

        self.processor = Processor()

    def _get_top_words(self, model, feature_names, n_top_words):
        topics = list()
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += ", ".join([feature_names[i]
                                  for i in topic.argsort()[:-n_top_words - 1:-1]])
            topics.append(message)

        return topics

    def _extract_token(self, text):
        docs = list(self.nlp.pipe(text))
        for index, doc in enumerate(docs):
            docs[index] = self.processor.extract_tokens_plus_meta(doc)

        return docs

    def apply_transformer(self, docs):
        cv = CountVectorizer(lowercase=False, ngram_range=(3, 4), analyzer="word", decode_error="ignore")
        cv.fit(docs)
        return cv, cv.get_feature_names_out()

    def _train_topic_model(self, X, n_components=5):
        nmf = NMF(n_components=n_components, max_iter=500)
        nmf.fit(X)
        return nmf

    def build_model(self, text, n_components=3, top_words=8):
        docs = self._extract_token(text)
        cvt, feature_names = self.apply_transformer(docs)
        X = cvt.transform(docs)
        nmf = self._train_topic_model(X, n_components)
        topics = self._get_top_words(nmf, feature_names, top_words)
        return topics



class ProductAnalyzer:
    def __init__(self):
        self.model = Modeller(load_small=True)

    def get_topics(self, data):
        return self.model.build_model(data)

