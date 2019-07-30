#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:31:37 2019
written by: Alan and Shazidul
tested by: Alan, Shazidul, Akshat, Anthony, Joel
debugged by: Alan and Shazidul

 If the flask does not run properly that may mean it is not installed into the python packages
 and if it still does not work then in the utils.py files change the file argument in the echo
 method to sys.stdout and do the same in in the secho method in the termui.py method. 
 After the flask is running open up the local server on any internet browser to see the webpage.
"""
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def output():
        return render_template('index.html')


    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../Demo1Code"))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

books = pd.read_csv('books.csv', encoding = "ISO-8859-1")
books.head()

ratings = pd.read_csv('ratings.csv', encoding = "ISO-8859-1")
ratings.head()

book_tags = pd.read_csv('book_tags.csv', encoding = "ISO-8859-1")
book_tags.head()

tags = pd.read_csv('tags.csv')
tags.tail()

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
tags_join_DF.head()

to_read = pd.read_csv('to_read.csv')
to_read.head()

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Build a 1-dimensional array with book titles
titles = books['title']
indices = pd.Series(books.index, index=books['title'])
"""
def authors_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

authors_recommendations('The Hobbit')
"""
books_with_tags = pd.merge(books, tags_join_DF, left_on='book_id', right_on='goodreads_book_id', how='inner')

tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf1.fit_transform(books_with_tags['tag_name'].head(10000))
cosine_sim1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)

# Build a 1-dimensional array with book titles
titles1 = books['title']
indices1 = pd.Series(books.index, index=books['title'])
"""
def tags_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]

tags_recommendations('The Hobbit').head(20)
"""
temp_df = books_with_tags.groupby('book_id')['tag_name'].apply(' '.join).reset_index()
temp_df.head()

books = pd.merge(books, temp_df, left_on='book_id', right_on='book_id', how='inner')
books.head()

books['corpus'] = (pd.Series(books[['authors', 'tag_name']]
                .fillna('')
                .values.tolist()
                ).str.join(' '))

tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)

titles = books['title']
indices = pd.Series(books.index, index=books['title'])
img = books['image_url']
indices2 = pd.Series(books.index, index=books['image_url'])


#this is the main function, the rest of functions for just author or just genre
def alpha_recommendations(title,title2,title3):
    idx = indices1[title]
    idx2 = indices1[title2]
    idx3 = indices1[title3]
    sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    #sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores2 = list(enumerate(cosine_sim_corpus[idx2]))
    sim_scores3 = list(enumerate(cosine_sim_corpus[idx3]))
    #sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    total = [(c, e+h) for (c, e), (d, h) in zip(sim_scores, sim_scores2)]
    total  = [(c, e+h) for (c, e), (d, h) in zip(total, sim_scores3)]
    #total = list( map(add, sim_scores, sim_scores2))
    #total = list( map(add, total, sim_scores3))
    total = sorted(total, key=lambda x: x[1], reverse=True)
    total = total[3:23]
    print ("tup1[0]: ", total[0])
    book_indices = [i[0] for i in total]
    return titles.iloc[book_indices]
    
import re
regex = re.compile('[^a-zA-Z, ()#0-9]')


@app.route('/result',methods = ['POST','GET'])
def  result():
    if request.method == 'POST':
        result = request.form.getlist('bookname')
        result = regex.sub('',str(result))
        b = result.split(',')
        #print(result)
        idx = indices1[b[0]]
        idx2 = indices1[b[1]]
        idx3 = indices1[b[2]]
        sim_scores = list(enumerate(cosine_sim_corpus[idx]))
    #sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores2 = list(enumerate(cosine_sim_corpus[idx2]))
    sim_scores3 = list(enumerate(cosine_sim_corpus[idx3]))
    #sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    total = [(c, e+h) for (c, e), (d, h) in zip(sim_scores, sim_scores2)]
    total  = [(c, e+h) for (c, e), (d, h) in zip(total, sim_scores3)]
    #total = list( map(add, sim_scores, sim_scores2))
    #total = list( map(add, total, sim_scores3))
    total = sorted(total, key=lambda x: x[1], reverse=True)
    #total = total[1:21]
    diff= total[3][1]-total[52][1]
    Lower= total[52][1]
    s=Lower+(diff*.95)
    a= Lower+(diff*.85)
    b= Lower+(diff*.6)
    c=Lower
    """The top portion is basically so that the tier 
    brackets work with any set of data. This means it doesn't matter how
    diverse the numbers are, every tier will have values, and it 
    will be in a bell curve model"""
    S=[]
    A=[]
    B=[]
    C=[]
    count= 4
    while count<53 and total[count][1]>=c:
        if total[count][1]>=b:
            while count<53 and total[count][1]>=b:
                if total[count][1]>=a:
                    while count<53 and total[count][1]>=a:
                        if total[count][1]>=s:
                            while count<53 and total[count][1]>=s:
                                S.append(total[count])
                                count+=1
                        else:
                            A.append(total[count])
                            count+=1
                else:
                    B.append(total[count])
                    count+=1
        else: 
            C.append(total[count])
            count+=1
    
    book_indicesS = [i[0] for i in S] 
    book_indicesA= [j[0] for j in A]
    book_indicesB= [k[0] for k in B]
    book_indicesC= [l[0] for l in C]
    tierS=titles.iloc[book_indicesS]
    tierSImg=img.iloc[book_indicesS]
    tierA=titles.iloc[book_indicesA]
    tierAImg=img.iloc[book_indicesA]
    tierB=titles.iloc[book_indicesB]
    tierBImg=img.iloc[book_indicesB]
    tierC=titles.iloc[book_indicesC]
    tierCImg=img.iloc[book_indicesC]
        # return str(abc)
        # return str(result)
        # return indices.astype('str')
    return render_template("result.html",result = tierSImg,result2 = tierAImg, result3 = tierBImg, result4 = tierCImg)
if __name__ == '__main__':
    app.run(debug = True)
#alpha_recommendations("The Hobbit", "The Catcher in the Rye", "Romeo and Juliet")

