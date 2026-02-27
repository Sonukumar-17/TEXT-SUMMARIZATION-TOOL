# text_summarizer.py

import nltk
import heapq
import re

# Download resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def summarize_text(text, summary_length=3):
    """
    Summarizes input text using frequency-based extraction.
    :param text: str, input article or paragraph
    :param summary_length: int, number of sentences in summary
    :return: str, summarized text
    """

    # Clean text
    text = re.sub(r'\s+', ' ', text)

    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w.isalnum() and w not in stop_words]

    # Word frequency
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    # Normalize frequencies
    max_freq = max(freq.values())
    for word in freq:
        freq[word] = freq[word] / max_freq

    # Sentence scores
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freq[word]

    # Select top sentences
    summary_sentences = heapq.nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)

    return summary


if __name__ == "__main__":
    # Example usage
    article = """
    Artificial Intelligence (AI) is transforming industries across the globe.
    From healthcare to finance, AI-powered tools are improving efficiency,
    accuracy, and decision-making. However, challenges such as bias,
    transparency, and ethical concerns remain. Researchers are working
    to make AI more trustworthy and accessible for everyone.
    """

    print("Original Text:\n", article)
    print("\nSummary:\n", summarize_text(article, summary_length=2))