# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 23:38:06 2019

@author: luosx14
"""
import functions
import os
from tqdm import tqdm
import re
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize

'''fit into matching pattern'''
def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list
C_stop_char = set(list('.?!。？！'))
def C_sentence_split(line):
    if len(line) < 2:
        return[]
    sen_list = []
    line = line.split('|||')[0]
    temp = ''
    for i in range(len(line)):
        w = line[i]
        temp += w
        if w == '.':
            if len(temp) > 1 and functions.is_number(temp[-2]) and (i < len(line) - 1) and functions.is_number(line[i+1]):
                continue
            if len(temp) > 1 and functions.is_number(temp[-2]) and (i < len(line) - 2) and functions.is_number(line[i+2]) and line[i+1] == ' ':
                continue
            if i < len(line) - 1 and line[i+1] != ' ':
                continue
        if w in C_stop_char:
            sen_list.append(temp)
            temp = ''
    return sen_list
select_fold = './version1_results/mapped_sentence/'
for file in functions.file_name(select_fold):
    C_map = {}
    E_map = {}
    with open('./split_CPM_data/'+file.split('.')[0]+'_C.snt','r',encoding='utf-8') as f:
        C_original = {}
        for line in f:
            C_original[line.strip()] = len(C_original)+1
    with open('./split_CPM_data/'+file.split('.')[0]+'_E.snt','r',encoding='utf-8') as f:
        E_original = {}
        for line in f:
            E_original[line.strip()] = len(E_original)+1
    with open(select_fold+file,'r',encoding='utf-8') as f:
        context = f.readlines()
        temp = 0
        C_line = {}
        E_line = {}
        for line in context:
            temp += 1
            C_list = C_sentence_split(line.split('|||')[0])
            E_list = sentence_token_nltk(line.split('|||')[1].strip('\n'))
            for line in C_list:
                for i in range(len(line)):
                    if line[i] not in ['\t',' ']:
                        break
                line = line[i:]
                
                if line not in C_original:
                    if line not in C_original:
                        print(file,line)
                C_line[line] = temp
                flag = True
                for o in C_original:
                    if o == line:
                        flag = False
                        if C_line[line] not in C_map:
                            C_map[C_line[line]] = [str(C_original[o])]
                        else:
                            C_map[C_line[line]].append(str(C_original[o]))
                if flag:
                    print(file,line)
            for line in E_list:
                if line[0] == ' ':
                    line = line[1:]
                E_line[line] = temp
                if line not in E_original:
                    print(line)
                flag = True
                for o in E_original:
                    if o == line:
                        flag = False
                        if E_line[line] not in E_map:
                            E_map[E_line[line]] = [str(E_original[o])]
                        else:
                            E_map[E_line[line]].append(str(E_original[o]))
                if flag:
                    print(file,line)
    with open('./version1_results/mapped_result/'+file.split('.')[0]+'.align','w',encoding='utf-8') as f:
        for i in range(len(C_map)):
            f.write(','.join(C_map[i+1])+':'+','.join(E_map[i+1])+'\n')



'''evaluation'''
manual_align = functions.file_name('./manual_align/')
selected = []
with open('random_choice_doc_list.txt','r') as f:
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 1:
            selected.append(line[0])
print(len(selected))
intersection = list(set.intersection(set(manual_align),set(selected)))
print(len(intersection))
def line_split(line,sep = ','):
    if len(line) == 0:
        return ['empty']
    else:
        return line.split(sep)

def get_CtoE_dict(file_dir):
    with open(file_dir,'r') as f:
        CtoE = {}
        for line in f:
            line = line.strip().split(':')
            for C_line in line_split(line[0],'no sep'):
                if C_line not in CtoE:
                    CtoE[C_line] = line_split(line[1])
                else:
                    CtoE[C_line].append(line_split(line[1]))
    return CtoE


def metric(MS_CtoE,manual_CtoE):
    '''only for 1-to-1 align'''
    MS_pre,MS_rec = [],[]
    for i in MS_CtoE:
        if i == 'empty' or ',' in i or len(MS_CtoE[i]) > 1 or MS_CtoE[i] == 'empty':
            continue
        if i in manual_CtoE and MS_CtoE[i] == manual_CtoE[i]:
            MS_pre.append(1)
        else:
            MS_pre.append(0)
            #print(file,i,j,MS_CtoE[i])
    for i in manual_CtoE:
        if i == 'empty' or ',' in i or len(manual_CtoE[i]) > 1 or manual_CtoE[i] == 'empty':
            continue
        if i in MS_CtoE and manual_CtoE[i] == MS_CtoE[i]:
            MS_rec.append(1)
        else:
            MS_rec.append(0)
            #print(file,i,j)
    return MS_pre,MS_rec


def multi_align(i,j,manual_CtoE):
    align_set = []
    for line in i.split(','):
        if i in manual_CtoE:
            align_set += manual_CtoE[i]
    if len(set(j).symmetric_difference(set(align_set))) == 0:
        return 1
    else:
        #print(i,j,manual_CtoE)
        return 0
    
def n_to_n_metric(our_CtoE,manual_CtoE):
    '''only for complement of 1-to-1 align'''
    MS_pre,MS_rec = [],[]
    for i in our_CtoE:
        if i == 'empty':
            continue
        j = our_CtoE[i]
        if j == 'empty' or ',' not in i+','.join(j):
            continue
        MS_pre.append(multi_align(i,j,manual_CtoE))
            #print(file,i,j,our_CtoE[i])
    for i in manual_CtoE:
        if i == 'empty':
            continue
        j = manual_CtoE[i]
        if j == 'empty' or  ',' not in i+','.join(j):
            continue
        MS_rec.append(multi_align(i,j,our_CtoE))
            #print(file,i,j,manual_CtoE[i])
    return MS_pre,MS_rec
    
def f1(BLEU_pre,BLEU_rec):
    pre = sum(BLEU_pre)/len(BLEU_pre)
    rec = sum(BLEU_rec)/len(BLEU_rec)
    return 2*pre*rec/(pre+rec)
manual_sum = 0
manual_1_to_1_sum = 0
MS_pre,MS_rec,our_pre,our_rec,GC_pre,GC_rec = [],[],[],[],[],[]
our_n_to_n_pre,our_n_to_n_rec = [],[]
for file in intersection:
    
    manual_CtoE = get_CtoE_dict('./manual_align/'+file)
    manual_sum += len(manual_CtoE)
    for i in manual_CtoE:
        if ',' in i or len(manual_CtoE[i]) > 1:
            continue
        else:
            manual_1_to_1_sum += 1
    our_CtoE = get_CtoE_dict('./version1_results/mapped_result/'+re.sub('.txt','.align',file))
    temp_pre,temp_rec = metric(our_CtoE,manual_CtoE)
    our_pre += temp_pre
    our_rec += temp_rec
    if len(temp_pre) > 0 and min(temp_pre) == 0:
        print(file)
    temp_pre,temp_rec = n_to_n_metric(our_CtoE,manual_CtoE)
    our_n_to_n_pre += temp_pre
    our_n_to_n_rec += temp_rec

print(len(our_pre),'precision:',sum(our_pre)/len(our_pre),len(our_rec),'recall:',sum(our_rec)/len(our_rec),'f1:',f1(our_pre,our_rec))
print(len(our_n_to_n_pre),'n_to_n_precision:',sum(our_n_to_n_pre)/len(our_n_to_n_pre),
      len(our_n_to_n_rec),'n_to_n_recall:',sum(our_n_to_n_rec)/len(our_n_to_n_rec),
      'f1:',f1(our_n_to_n_pre,our_n_to_n_rec))

