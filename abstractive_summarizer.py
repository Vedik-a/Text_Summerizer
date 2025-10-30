from transformers import pipeline

_summarizer = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer

def abstractive_summary(text, min_length=30, max_length=120):
    summarizer = _get_summarizer()
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']
