"""
    author: alan hamm
    modified: https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3

    date march 2024
"""
#%%

from pprint import pprint 
import io
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import pprint 
import io
import re

#%%
ff = io.open(r'C:\Users\pqn7\OneDrive - CDC\a.Reference.Documentation\GenAIGuidance-processed.txt', 'r', encoding='latin1')
doc=ff.read()
ff.close()


#%%
from time import time
start = time()
for line in doc:
    sentence = " ".join(line for line in doc.split("\n"))
pprint.pprint(len(sentence))
#pprint(sentences)

finish_time = round((time() - start)/60, 2)
pprint.pprint(f"Time to complete {finish_time}")


#%%
pprint.pprint(sentence)


#%%
'''
We already have a sentence tokenizer, so we just need 
to run the sent_tokenize() method to create the array of sentences.
'''
# 1 Sentence Tokenize
sentences = sent_tokenize(sentence)
total_documents = len(sentence)
pprint.pprint(total_documents)


#%%
""" Create the Frequency matrix of the words in each sentence """
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


#%%
""" Calculate TermFrequency and generate a matrix """
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


#%%
""" Create a table for documents per words """
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


#%%
""" Calculate IDF and generate a matrix """
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


#%%
""" Calculate TF-IDF and generate a matrix """
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


#%%
""" Score the sentences """
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


#%%
""" Find the threshold """
def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


#%%
""" generate the summary """
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


#%%
# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentence)
pprint.pprint(freq_matrix)

#%%
'''
Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
'''
# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
pprint.pprint(tf_matrix)


#%%
# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
pprint.pprint(count_doc_per_words)

#%%
'''
Inverse document frequency (IDF) is how unique or rare a word is.
'''
# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
print(idf_matrix)

#%%
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
pprint.pprint(tf_idf_matrix)

#%%

# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
pprint.pprint(sentence_scores)

#%%



# 8 Find the threshold
threshold = _find_average_score(sentence_scores)
pprint.pprint(threshold)

#%%
# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
pprint.pprint(summary)

# %%