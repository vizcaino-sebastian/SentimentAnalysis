# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:59:55 2021

@author: vizcaino-j
"""
import pandas as pd
# Classes provided from AdvancedAnalytics

from AdvancedAnalytics.Text import text_analysis
from AdvancedAnalytics.Text import sentiment_analysis
from sklearn.feature_extraction.text import CountVectorizer # Trouble 

# AFINN List - 
# You can use the Google CLOUD API. FOR diff. Version ( look for google json files)
#  -- Go to GoogleNLP1.py for more on that. 
pd.set_option('max_colwidth', 32000)
df = pd.read_excel('PythonTestCases.xlsx')

df['SENTIMENT'] = df['SENTIMENT'].map('{:,.2f}'.format)
print(df)

text_col = "TEXT"
pd_width = pd.get_option('max_colwidth')
maxsize = df[text_col].map(len).max()
n_truncated = (df[text_col].map(len)>pd_width).sum()
print("\nTEXT LENGTH: ")
print("{:<17s}{:>6d}".format(" Max. Accepted", pd_width))
print("{:<17s}{:>6d}".format(" Max. Observed", maxsize))
print("{:<17s}{:>6d}".format(" Truncated", n_truncated))


ta = text_analysis(synonyms=None, stop_words=None, pos=False, stem=False)

# n_terms=2 only displays text containing 2 or more sentiment words for
# the list of the highest and lowest sentiment strings
sa = sentiment_analysis(n_terms=2)


# Create Word Frequency by Review Matrix using Custom Sentiment
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
ngram_range=(1,2), analyzer=sa.analyzer, \
vocabulary=sa.sentiment_word_dic)
stf = cv.fit_transform(df[text_col])
sterms = cv.get_feature_names()
# Calculate and Store Sentiment Scores into DataFrame "s_score"
s_score = sa.scores(stf, sterms)


n_reviews = s_score.shape[0]
n_sterms = s_score['n_words'].sum()
max_length = df['TEXT'].apply(len).max()
if n_sterms == 0 or n_reviews == 0:
    print("No sentiment terms found.")

p = s_score['n_words'].sum() / n_reviews
print('{:−<24s}{:>6d}'.format("\nMaximum Text Length", max_length))
print('{:−<23s}{:>6d}'.format("Total Reviews", n_reviews))
print('{:−<23s}{:>6d}'.format("Total Sentiment Terms", n_sterms))
print('{:−<23s}{:>6.2f}'.format("Avg. Sentiment Terms", p))