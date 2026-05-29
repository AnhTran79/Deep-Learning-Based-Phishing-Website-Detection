from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logreg(max_features: int = 50000) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(3, 5))),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
