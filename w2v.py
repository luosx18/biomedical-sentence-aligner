# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 23:38:06 2019

@author: luosx14
"""

import gensim
from gensim.models import word2vec
import time

start = time.time()
filename = 'Bi_w2v_300_win40_sg1_1Repeats_Cpubmed'
corpus_C=word2vec.Text8Corpus('./data cut all false/w2v_proportion_insert.txt')
model_C = word2vec.Word2Vec(corpus_C, size=300,window = 10,sg = 1,workers=4,min_count=3)
model_C.wv.save_word2vec_format("./model_save/"+filename+".bin",binary=True)
end = time.time()
print(end-start)
