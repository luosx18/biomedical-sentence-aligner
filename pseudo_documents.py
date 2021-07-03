# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:08:02 2019

@author: luosx14
"""

import jieba
import os
import re
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
        else:
                return False
def is_chinese_word(s):
    if len(s) == 0:
        return False
    else:
        for w in s:
            if is_chinese(w) == False:
                return False
        return True
def is_number(uchar):
        """判断一个unicode是否是数字"""
        if uchar >= u'\u0030' and uchar<=u'\u0039':
                return True
        else:
                return False
def is_chinese_word_add_number(s):
    """中文混数字"""
    if len(s) == 0:
        return False
    else:
        for w in s:
            if is_chinese(w) == False and is_number(w) == False:
                return False
        return True
def is_alphabet(uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
                return True
        else:
                return False
def is_other(uchar):
        """判断是否非汉字，数字和英文字符"""
        if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
                return True
        else:
                return False
def has_other(s):
    """判断是否存在非汉字，数字和英文的字符"""
    for w in s:
        if is_other(w):
            return True
    return False
def is_english_word(ustr):
    if len(ustr) == 0:
        return False
    for char in ustr:
        if is_alphabet(char) == False:
            return False
    return True
def is_english_word_add_number(ustr):
    if len(ustr) == 0:
        return False
    for char in ustr:
        if is_alphabet(char) == False and is_number(char) == False:
            return False
    return True
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(file)
    return L
"""
jieba load lexicon and cut_all, if there's no lexicon, skip this part
"""
word_fre = 100
with open('./data/complete_lexicon_drop_duplicates.txt','r',encoding = 'utf-8') as f:
    C_lexicon = f.readlines()
with open('./data/C_lexicon.txt','w',encoding = 'utf-8') as f:
    for line in C_lexicon:
        f.write(line.strip('\n')+' '+str(word_fre)+'\n')
jieba.load_userdict('./data/C_lexicon.txt')

def jieba_cut_document(context,the_file_name):
    with open('./data cut all false/C_document_jieba_cut_all_false/' + the_file_name,'w',encoding = 'utf-8') as f:
        for line in context:
            line = line.strip('\n')
            f.write(' '.join(jieba.cut(line,cut_all = False)))
    
title_dir = "./original_pages/paper_title";text_dir = "./original_pages/C_abstract/"
text_file_list = file_name(file_dir=text_dir)


for file in tqdm(text_file_list):
    with open(text_dir + file, 'r', encoding='utf-8') as f:
        context = f.readlines()
        jieba_cut_document(context,file)
            
            
"""
count C_words frequency
"""
C_dist_count = {}
C_text_C_rate = {}
for file in tqdm(file_name("./data cut all false/C_document_jieba_cut_all_false/")):
    C_char = 0;C_len = 0
    with open('./data cut all false/C_document_jieba_cut_all_false/' + file, 'r', encoding='utf-8') as f:
        context = f.readlines()
        C_text_C_rate[file] = 'empty'
        if len(context) > 0:
            context = context[0].split(' ')
            for w in context:
                if len(w) > 0 and is_chinese_word_add_number(w):
                    if w not in C_dist_count:
                        C_dist_count[w] = 1
                    else:
                        C_dist_count[w] += 1
                    C_len += len(w)
                    for char in w:
                        if is_chinese(char):
                            C_char += 1
            C_text_C_rate[file] = C_char/C_len
np.save('./data cut all false/C_dist_count.npy',C_dist_count)
np.save('./data cut all false/C_text_C_rate.npy',C_text_C_rate)
C_dist_count = np.load('./data cut all false/C_dist_count.npy',allow_pickle=True).item()
C_text_C_rate = np.load('./data cut all false/C_text_C_rate.npy',allow_pickle=True).item()
"""
load umls and preprocess Etext
"""
Eterm_cui = np.load('./data/Eterm_cui.npy',allow_pickle=True).item()
cui_Eterm = np.load('./data/cui_Eterm.npy',allow_pickle=True).item()

def Etext_process(context,the_file_name):#all matched → cui
    with open('./data cut all false/Etext_process/' + the_file_name,'w',encoding = 'utf-8') as f:
        for line in context:
            line = re.sub('\\n',' ',line)
            temp = ''
            for i in range(len(line)):
                char = line[i]
                if is_alphabet(char) or is_number(char) or char in "-":
                    temp += char
                elif char == "'" and line[i+1] == 's':
                    temp += char
                else:
                    temp += ' '
            line = re.sub('\s+',' ',temp)
            line = line.lower()
            f.write(line)
for file in tqdm(file_name('./original_pages/E_abstract/')):
    with open("./original_pages/E_abstract/" + file, 'r', encoding='utf-8') as f:
        context = list(f.readlines())
        if len(context) > 0:
            Etext_process(context,file)

def match_all_cui(s,max_len = 10, Eterm_cui = Eterm_cui):
    """Forward Maximum Matching cui and delect matched words"""
    if len(s) == 0: 
        return []
    sub_label = np.zeros(len(s),dtype = 'int')
    location_term = {}
    i = 0
    while i < len(s):
        for j in range(max_len+1,0,-1):
            temp = ' '.join(s[i:i+j])
            if temp in Eterm_cui:
                sub_label[i:i+j] = 1
                location_term[i] = [Eterm_cui[temp]]
                break#matched maximum string, so break
        i += j
    output = []
    for i in range(len(s)):
        if sub_label[i] == 0:#no match
            output += [s[i]]
        elif i in location_term:
            for cui in location_term[i][: :-1]:
                output += [cui]
    return output

for file in tqdm(file_name('./data cut all false/Etext_process/')):
    with open('./data cut all false/Etext_process/' + file, 'r', encoding='utf-8') as f:
        context = ''
        for line in f.readlines():
            context += line.strip('\n')
        
        if len(context) > 0:
            context = context.split(' ')
            output = match_all_cui(context,max_len = 10, Eterm_cui = Eterm_cui)
    with open('./data cut all false/Etext_process_max_match_only/' + file, 'w', encoding='utf-8') as f:
        f.write(' '.join(output))

E_dist_count = {}
for file in tqdm(file_name('./data cut all false/Etext_process_max_match_only')):
    with open('./data cut all false/Etext_process_max_match_only/' + file, 'r', encoding='utf-8') as f:
        context = f.readlines()
        if len(context) > 0:
            context = context[0].split(' ')
            for w in context:
                if is_english_word_add_number(w):
                    if w not in E_dist_count:
                        E_dist_count[w] = 1
                    else:
                        E_dist_count[w] += 1
np.save('./data cut all false/E_dist_count.npy',E_dist_count)
E_dist_count = np.load('./data cut all false/E_dist_count.npy',allow_pickle=True).item()




"""
生成pseudo txt
"""
C_dist_count = np.load('./data cut all false/C_dist_count.npy',allow_pickle=True).item()
C_text_C_rate = np.load('./data cut all false/C_text_C_rate.npy',allow_pickle=True).item()
"""
生成pseudo txt
"""
def pseudo_document(C_context,E_context):
    temp = []
    C_len = len(C_context);E_len = len(E_context)
    while C_len > 0 or E_len > 0:
        total = C_len + E_len
        C_flag = np.random.choice([True,False], size=1, replace=True, p=[C_len/total,E_len/total])
        if C_flag:
            if is_chinese_word_add_number(C_context[-C_len]):
                temp += [C_context[-C_len]]
            C_len -= 1
        else:
            temp += [E_context[-E_len]]
            E_len -= 1
    return temp

def relative_location(length):
    a = np.linspace(0,length-1,length,dtype = 'int')
    b = np.linspace(1,length,length,dtype = 'int')
    return (a+b)/2/length
def both_relative_location(C_len,E_len):
    return relative_location(C_len),relative_location(E_len)
def proportion_insert(C_context,E_context):
    temp = []
    C_len = len(C_context);E_len = len(E_context)
    C_relative_location,E_relative_location = both_relative_location(C_len,E_len)
    while C_len > 0 or E_len > 0:
        if (C_relative_location[-C_len] < E_relative_location[-E_len] and C_len > 0) or E_len <= 0:
            if is_chinese_word_add_number(C_context[-C_len]):
                temp += [C_context[-C_len]]
            C_len -= 1
        else:
            temp += [E_context[-E_len]]
            E_len -= 1
    return temp
def C_text_clean(C_context):
    temp = []
    for w in C_context:
        if is_chinese_word(w):
            temp.append(w)
    return temp
def E_text_clean(E_context):
    temp = []
    for w in E_context:
        if len(w) > 0 and is_alphabet(w[0]):
            temp.append(w)
    return temp

pseudo_file = open('./data cut all false/w2v_proportion_insert.txt', 'a+', encoding='utf-8')

for repeat in range(1):
    step = 0
    for file in tqdm(file_name('./data cut all false/Etext_process_max_match_only')):
        if type(C_text_C_rate[file]) == float and C_text_C_rate[file] > 0:
            with open('./data cut all false/C_document_jieba_cut_all_false/' + file, 'r', encoding='utf-8') as f:
                context = f.readlines()
            C_context = C_text_clean(context[0].split(' '))
            if '引言' in C_context[0]:
                C_context[0] = C_context[0].strip('引言')
                C_context.insert(0,'引言')
            with open('./data cut all false/Etext_process_max_match_only/' + file, 'r', encoding='utf-8') as f:
                context = f.readlines()
            E_context = E_text_clean(context[0].split(' '))
            if len(C_context) == 0 or len(E_context) == 0:
                continue
            output = proportion_insert(C_context,E_context)
            pseudo_file.write(' '.join(output)+'\n')
            step += 1
pseudo_file.close()



