from matplotlib import pyplot as plt
from nltk.tokenize import regexp_tokenize
import re

def docLineWordsHist(doc):
    lines = doc.split('\n')

    # substitute line headers
    pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
    lines = [re.sub(pattern, '', line) for line in lines]

    # split each line into words
    tokenized_lines = [regexp_tokenize(line, "\w+") for line in lines]

    # counts of words for each line
    line_num_words = [len(line_words_array) for line_words_array in tokenized_lines]

    plt.hist(line_num_words)
    plt.show()