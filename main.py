from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Extracteurs rapides
class TextJoin(BaseEstimator, TransformerMixin):
    def __init__(self, keys): self.keys = keys
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([" ".join([x.get(k,"") or "" for k in self.keys]) for x in X])

class NumPick(BaseEstimator, TransformerMixin):
    def __init__(self, keys): self.keys = keys
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([[x.get(k,0) or 0 for k in self.keys] for x in X], dtype=float)

class CatPick(BaseEstimator, TransformerMixin):
    def __init__(self, key): self.key = key
    def fit(self, X, y=None): return self
    def transform(self, X):
        return np.array([[x.get(self.key,"UNK")] for x in X])

text = ('text', HashingVectorizer(ngram_range=(1,2), n_features=2**20, alternate_sign=False),
        TextJoin(['subject','from_domain','snippet']))
num = ('num', 'passthrough', NumPick([
    'has_list_id','has_unsubscribe','thread_reply_count',
    'recv_30d_count_sender','days_since_first_seen_sender',
    'size_kb','subject_len','is_calendar','is_finance_keyword','is_ticket_keyword'
]))
cat = ('cat', OneHotEncoder(handle_unknown='ignore'),
       CatPick('gmail_category'))

pre = ColumnTransformer([text, num, cat], sparse_threshold=0.3)
clf = SGDClassifier(loss="log_loss", class_weight={1:3,0:1})

# Entraînement initial avec labels LLM haute confiance
X0, y0 = preload_from_llm()  # listes de dicts et labels 0/1
clf.partial_fit(pre.fit_transform(X0), y0, classes=np.array([0,1]))

def predict_proba_batch(msgs):
    X = pre.transform(msgs)
    return clf.predict_proba(X)[:,1]
