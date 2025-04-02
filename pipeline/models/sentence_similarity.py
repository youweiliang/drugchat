import time
import nltk
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import pandas as pd
import numpy as np
# from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer

import warnings
warnings.filterwarnings("ignore")

# Ensure that the necessary nltk resources are downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# word_freq = pd.read_csv('eval_results/similarity_scores/unigram_freq.csv')
# freq_words = set(word_freq['word'][:3000].tolist())

def contains_alpha(s):
    # Check if any character in the string is an alphabet letter
    return any(char.isalpha() for char in s)


def compute_meteor_score(reference, candidate):
    """
    Compute the METEOR score for a single candidate sentence against a reference.
    Args:
    reference (str): The reference sentence.
    candidate (str): The candidate sentence.
    Returns:
    float: The METEOR score.
    """
    # Tokenize the reference and candidate sentences
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    # Compute the METEOR score
    score = meteor_score([reference_tokens], candidate_tokens)
    return score


class SimilarityModel:
    def __init__(self, device) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        # self.model = SentenceTransformer('kamalkraj/BioSimCSE-BioLinkBERT-BASE', device=device)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    def calculate_similarity(self, target, prediction):

        # Compute embedding for both lists
        embeddings = self.model.encode([target, prediction])
        embeddings = embeddings / np.linalg.norm(embeddings, 2, axis=1, keepdims=True)

        cos_sim = (embeddings[0] * embeddings[1]).sum().item()

        # BLEU scores
        bleus = {}
        bleu_args = [target.split()], prediction.split()
        for i in range(4):
            weights = [0, 0, 0, 0]
            weights[i] = 1
            bleu = sentence_bleu(*bleu_args, weights=weights)
            bleus[f'bleu{i + 1}'] = bleu
        bleus['bleu'] = sum(bleus.values()) / 4

        meteor = compute_meteor_score(target, prediction)

        rouge_scores = self.rouge_scorer.score(target, prediction)
        out = {
            'cos_sim': cos_sim,
            'meteor': meteor,
            'rouge1_precision': rouge_scores['rouge1'].precision,
            'rouge1_recall': rouge_scores['rouge1'].recall,
            'rouge1_fmeasure': rouge_scores['rouge1'].fmeasure,
        }
        return out | bleus


# model = SimilarityModel('cuda')


if __name__ == "__main__":
    # nltk.download('punkt')
    # nltk.download('wordnet')
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SimilarityModel(3)
    out = model.calculate_similarity(sentences[0], sentences[1])
    print(out)