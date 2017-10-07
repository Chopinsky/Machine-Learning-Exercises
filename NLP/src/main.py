from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.book import text6

# pattern to match
match_digits_and_words = ('(\d+|\w+)')

# test cases
test1 = 'He has 11 cats'
test2 = 'This is a pretty cool tool!'
test3 = 'a b c\nd e f\nab cd ef\nabc def'

if __name__ == '__main__':
    import func

    #res = re.findall(match_digits_and_words, test1)

    words = word_tokenize(test2)
    word_len = [len(w) for w in words]

    plt.hist(word_len)
    plt.show()

    # call imported function
    func.docLineWordsHist(test3)

    print(text6.collocations())

    print("Done!")
