'''
script to generate tf-idf of words(s) in documents 

author alan hamm(pqn7@cdc.gov)
date september 2021
'''

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import pprint 
import io

#create PrettyPrinter instance
pp = pprint.PrettyPrinter(indent=1)

#read text document 
ff = io.open(r'\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP\incomeDoc.txt', 'r', encoding='utf-8')
doc=ff.read()
ff.close()

#tokenize document
sentences = sent_tokenize(doc)
'''
pp.pprint(sentences)
for sent in sentences:
    print(sent)
    print('')
'''

total_documents = len(sentences)
#print(total_documents)

#get count of words in each sentence
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

        frequency_matrix[sent[:100]] = freq_table

    return frequency_matrix

#freq_matrix = _create_frequency_matrix(sentences)
#pp.pprint(freq_matrix)


#get term frequency
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

#tf_matrix = _create_tf_matrix(freq_matrix)
#pp.pprint(tf_matrix)


#get number of documents containing a term
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

#term_doc = _create_documents_per_words(freq_matrix)
#pp.pprint(term_doc)


#create IDF 
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

#idf_matrix = _create_idf_matrix(freq_matrix, term_doc, total_documents)
#pp.pprint(idf_matrix)


#create tf-idf matrix
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

#tf_idf = _create_tf_idf_matrix(tf_matrix, idf_matrix)
#pp.pprint(tf_idf)


#get relevancy of sentence in a document
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

#sent_score = _score_sentences(tf_idf)
#pp.pprint(sent_score)


#get average sentence score -> the threshold
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

#threshold = _find_average_score(sent_score)
#pp.pprint(threshold)


#get summary of sentence if its score is greater than average sentence score
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:100] in sentenceValue and sentenceValue[sentence[:100]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

#summary = _generate_summary(sentences, sent_score, threshold)
#pp.pprint(summary)



''' step one '''
freq_matrix = _create_frequency_matrix(sentences)
#pp.pprint(freq_matrix)

''' step two '''
tf_matrix = _create_tf_matrix(freq_matrix)
#pp.pprint(tf_matrix)

''' step three '''
term_doc = _create_documents_per_words(freq_matrix)
#pp.pprint(term_doc)

''' step four '''
idf_matrix = _create_idf_matrix(freq_matrix, term_doc, total_documents)
#pp.pprint(idf_matrix)

tf_idf = _create_tf_idf_matrix(tf_matrix, idf_matrix)
#pp.pprint(tf_idf)

''' step five '''
sent_score = _score_sentences(tf_idf)
#pp.pprint(sent_score)

'''step six '''
threshold = _find_average_score(sent_score)
#pp.pprint(threshold)

'''step seven '''
summary = _generate_summary(sentences, sent_score, threshold)
#pp.pprint(summary)
#print(sentences)
#print('')
print(summary)