from rouge import Rouge

def evaluate_summary(reference, generated):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return scores
