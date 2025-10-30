import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _ensure_nltk_tokenizers():
    """Ensure NLTK resources needed for sentence tokenization are available.

    Newer NLTK versions split resources into 'punkt' and 'punkt_tab'. We try to
    locate them and download only if missing, to avoid repeated downloads.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


def extractive_summary(text, num_sentences=3):
    if not text or not str(text).strip():
        return ""
    _ensure_nltk_tokenizers()
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf, tfidf)
    scores = similarity_matrix.sum(axis=1)
    top_indices = np.argsort(scores)[::-1][:num_sentences]
    top_indices_sorted = sorted(top_indices)
    summary = " ".join([sentences[i] for i in top_indices_sorted])
    return summary
