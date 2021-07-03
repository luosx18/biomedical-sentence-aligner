# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:35:47 2019

@author: luosx14
"""


"""
需要几个函数：
1、中英文句子切分器，需要返回句子以及句子在文章中的相对位置以及句子本身(as [])的相对长度
2、计算句子在文本中的相对位置的函数
3、句子相对长度的函数
4、度量两个句子的相似性的函数
"""
import functions
import os
from tqdm import tqdm
import re
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize


def line_process(line):
    line = re.sub('[•●\\n]','',line)
    return line
def context_len(context):
    length = 0
    for line in context:
        length += len(line)
    return length
C_stop_char = set(list('.?!。？！'))
def C_sentence_spliter(context,C_stop_char = C_stop_char):
    sen_list = []
    posi = []
    len1 = []
    length_all = context_len(context)
    if length_all == 0:
        return [],[],[]
    length_now = 0
    added_len = 0
    for line in context:
        temp = ''
        for i in range(len(line)):
            w = line[i]
            temp += w
            length_now += 1
            if w == '.':
                """特殊对待英文句号，因为它可能是小数点,如果是 0.4 or 0. 4 这种情况，则continue !!!实际发现.还可能是网址......"""
                if len(temp) > 1 and functions.is_number(temp[-2]) and (i < len(line) - 1) and functions.is_number(line[i+1]):
                    continue
                if len(temp) > 1 and functions.is_number(temp[-2]) and (i < len(line) - 2) and functions.is_number(line[i+2]) and line[i+1] == ' ':
                    continue
                if i < len(line) - 1 and line[i+1] != ' ':
                    continue
            if w in C_stop_char:# or i == len(line)-1:#是停止位置或者是一行的结束
                """目前没有把一句话终止没有终止符的情况考虑进去，由于这样会产生大量的干扰"""
                """没有处理一对括号中有终止符导致最终出现括号不完整的情况"""
                sen_list.append(temp);added_len += len(temp)
                posi.append((length_now-len(temp)/2)/length_all)
                len1.append(len(temp))
                temp = ''
    len1 = np.array(len1)/added_len
    return sen_list,posi,len1
def E_sentence_spliter(context):
    sen_list = []
    posi = []
    len1 = []
    length_all = context_len(context)
    if length_all == 0:
        return [],[],[]
    length_now = 0
    for line in context:
        sen_list += sentence_token_nltk(line)
    length_all = context_len(sen_list)
    for line in sen_list:
        length_now += len(line)
        len1.append(len(line))
        posi.append((length_now-len(line)/2)/length_all)
    len1 = np.array(len1)/length_all
    return sen_list,posi,len1





"""
derive three distance
"""
import gensim
filename = 'Bi_w2v_300_win40_sg1_proportion_insert_Cpubmed'
model = gensim.models.KeyedVectors.load_word2vec_format('./model_save/'+filename+".bin", binary=True)
E_dist_count = np.load('./data cut all false/E_dist_count.npy',allow_pickle=True).item()
C_dist_count = np.load('./data cut all false/C_dist_count.npy',allow_pickle=True).item()

import jieba
jieba.load_userdict('./data/C_lexicon.txt')

Eterm_cui = np.load('./data/Eterm_cui.npy',allow_pickle=True).item()
cui_Eterm = np.load('./data/cui_Eterm.npy',allow_pickle=True).item()
def max_match_cui(s,max_len = 10, Eterm_cui = Eterm_cui):
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
def Etext_process(line):#all matched → cui
    punctuation = set(list('.,?;\\'))
    line = re.sub('\\n',' ',line)
    temp = ''
    for i in range(len(line)):
        char = line[i]
        if functions.is_alphabet(char) or functions.is_number(char) or char in "-":
            temp += char
        elif char == "'" and i < len(line) -1 and line[i+1] == 's':
            temp += char
        elif char in punctuation:
            temp += ' ' + char + ' '
        else:
            temp += ' '
    line = re.sub('\s+',' ',temp)
    line = line.lower()
    return line
def cos_sim(v1,v2):
    return np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    
def sim(w,t,n = 1):
    w_vec = np.array(model[w])
    sim_dist = {}
    flag = True
    for t_w in t:
        if t_w in model:
            sim_dist[t_w] = cos_sim(w_vec,model[t_w])
            flag = False
    if flag:
        return [[0,0.0001]]
    return sorted(sim_dist.items(),key=lambda x:-x[1])[:n]
def get_all_word_to_line_dict(sen_list,language = 'C'):
    w2l = {}
    for i in range(len(sen_list)):
        line = sen_list[i]
        if language == 'C':
            line = list(jieba.cut(line,cut_all = False))
        else:
            line = word_tokenize(line)
        for w in line:
            if functions.is_number_str(w):
                if w in w2l:
                    w2l[w].add(i)
                else:
                    w2l[w] = set([i])
    return w2l
        
def in_other_s_sen(w,w2l):
    if w in w2l:
        if len(w2l[w]) > 1:
            return True
    return False
def only_in_t_j(w,t_j,E_w2l):
    if w in E_w2l:
        if len(E_w2l[w]) == 1 and t_j in E_w2l[w]:
            return True
    return False
    
def max_match_similar(w,t_j,C_w2l,E_w2l):#t in j_th line
    if in_other_s_sen(w,C_w2l) == False and only_in_t_j(w,t_j,E_w2l):
        return True
    return False
    
def d1(si,tj,len_s,len_t,model,t_j,C_w2l,E_w2l):
    s = list(jieba.cut(si,cut_all = False))
    t = max_match_cui(re.split('\s+',Etext_process(tj)),max_len = 10, Eterm_cui = Eterm_cui)
    sum_sim = 0.01
    dist1 = 100

    temp_len_s = 0
    for w in s:
        if w not in C_dist_count:
            continue
        if w in model:
            temp_len_s += 1
            sum_sim += sim(w,t)[0][1]
        else:#w not in model
            if max_match_similar(w,t_j,C_w2l,E_w2l):
                temp_len_s += 1
                sum_sim += 5.0
        dist1 = 10*temp_len_s/sum_sim
    return dist1
def d2(len_s,len_t):
    return np.linalg.norm(len_s-len_t)
def d3(posi_s,posi_t,threshold = 0.25):
    dis = np.linalg.norm(posi_s-posi_t)
    if dis < threshold:
        return 0
    return 10 * dis
def d(i,j,C_sen_list,E_sen_list,C_len,E_len,C_posi,E_posi,C_w2l,E_w2l,alpha =  0,beta = 0, threshold = 0.25):
    si = C_sen_list[i];len_s = C_len[i];posi_s = C_posi[i]
    tj = E_sen_list[j];len_t = E_len[j];posi_t = E_posi[j]
    return d1(si,tj,len_s,len_t,model,j,C_w2l,E_w2l)+alpha*d2(len_s,len_t)+beta*d3(posi_s,posi_t,threshold)

def C_process(C_file_name):
    with open('./original_pages/'+C_file_name,'r',encoding='utf-8') as f:
        context = []
        for line in f:
            context.append(line_process(line))
    C_sen_list,C_posi,C_len = C_sentence_spliter(context)
    return C_sen_list,C_posi,C_len
def E_process(E_file_name):
    with open('./original_pages/'+E_file_name,'r',encoding='utf-8') as f:
        context = []
        for line in f:
            context.append(line_process(line))
    E_sen_list,E_posi,E_len = E_sentence_spliter(context)
    return E_sen_list,E_posi,E_len
from scipy import optimize
def get_z_A(C_file_name,E_file_name):
    C_sen_list,C_posi,C_len = C_process(C_file_name)
    E_sen_list,E_posi,E_len = E_process(E_file_name)
    C_w2l = get_all_word_to_line_dict(C_sen_list,language = 'C')
    E_w2l = get_all_word_to_line_dict(E_sen_list,language = 'E')
    
    N = len(C_sen_list);M = len(E_sen_list)
    if N * M <= 0:
        return 0,0,0,0,0,0
    dist = np.zeros((N,M));z = []
    for i in range(N):
        for j in range(M):
            dist[i,j] = d(i,j,C_sen_list,E_sen_list,C_len,E_len,C_posi,E_posi,C_w2l,E_w2l,
                0,1,threshold=3/np.mean([N,M]))
            z.append(1/dist[i,j])
    z = -np.array(z)
    A = np.zeros((N+M+1,N*M))
    for i in range(N):
        for j in range(M):
            A[i,M*i+j] = 1
    for j in range(M):
        for i in range(N):
            A[j+N,M*i+j] = 1
    A[N+M,:] = 1
    return z,A,N,M,C_len,E_len
    
def with_slack(z,A,N,M,C_len,E_len,epsilon = 0.1):
    epsilon1 = epsilon/N;epsilon2 = epsilon/M
    b = np.append(C_len+epsilon1,E_len+epsilon2)
    b = np.append(b,np.array(1.0))
    res = optimize.linprog(z,A,b,bounds = ((0.0,None),)*N*M)
    P = np.zeros((N,M))
    step = 0
    for i in range(N):
        for j in range(M):
            if res.x[step] < 10**-8:
                P[i,j] = 0.0
            else:
                P[i,j] = res.x[step]
            step += 1
    return P
def inconsistence(square):
    nonzero = square[square != 0]
    if nonzero.size <= 2:
        return 0
    else:
        return np.min(nonzero)
def inconsistence2(two_):
    inc2 = 0.0
    continue_flag = True
    if two_.shape[0] < two_.shape[1]:
        two_ = np.transpose(two_)
    while continue_flag:
        inc2_flag = False;continue_flag = False
        first_nonzero = 0;first_flag = True;second_flag = True
        first_row = 0
        i = 0
        while i < two_.shape[0]:
            if first_flag or second_flag:
                for j in range(two_.shape[1]):
                    if first_flag and two_[i,j] > 0:
                        first_flag = False
                        first_nonzero = j
                        first_row = i
                    elif first_flag == False and j != first_nonzero:
                        if two_[i,j] > 0:
                            second_flag = False
                            if first_row != i:
                                i -= 1
            else:
                if two_[i,first_nonzero] > 0:
                    inc2_flag = True
                    continue_flag = True
            i += 1
        if inc2_flag:
            nonzero = two_[two_ != 0]
            inc2 += np.min(nonzero)
            i,j = np.where(two_ == np.min(nonzero))
            two_[i[0],j[0]] = 0
    return inc2                  
def sum_inconsistence2(P):
    sum_inc2 = 0.0
    for i in range(P.shape[1]-1):
        two_ = np.copy(P[:,i:i+2])
        inc2 = inconsistence2(two_) 
        sum_inc2 += inc2
    for i in range(P.shape[0]-1):
        two_ = np.copy(P[i:i+2,:])
        inc2 = inconsistence2(two_)
        sum_inc2 += inc2
    return sum_inc2

    




def get_epsilon(all_inc2,all_epsilon,all_P,beta = 1):
    func = all_inc2 + beta * all_epsilon
    min_f = np.min(func)
    min_posi = np.min(np.where(func == min_f))
    min_e = all_epsilon[min_posi]
    P = all_P[min_posi]
    return min_e,P



def clean_P(P,N,M,threshold = 0.25):
    P_clean = P.copy()
    for i in range(N):
        for j in range(M):
            if P_clean[i,j] == 0:
                continue
            else:
                row_max = np.max(P_clean[i,:])
                col_max = np.max(P_clean[:,j])
                if P_clean[i,j] < threshold*np.max([row_max,col_max]):
                    P_clean[i,j] = 0
    return P_clean
def get_mapping(P_clean,N,M):
    S = {}
    T = {}
    for i in range(N):
        for j in range(M):
            if P_clean[i,j] > 0:
                if i not in S:
                    S[i] = set([j])
                else:
                    S[i].add(j)
                if j not in T:
                    T[j] = set([i])
                else:
                    T[j].add(i)
    return S,T
def set_mapping(S_map,T_map):
    S_has_map = set();#T_has_map = set()
    S_set,T_set = [],[]
    for i in range(N):
        if i in S_has_map or i not in S_map:
            continue
        S_has_map.add(i)
        temp_S,temp_T = set([i]),set()
        flag = True
        new_j = set()
        while flag:
            flag = False
            for j in set.union(S_map[i],new_j):
                if j not in temp_T:
                    flag = True
                    temp_T.add(j)
                    for si in T_map[j]:
                        if si not in temp_S:
                            temp_S.add(si)
                            S_has_map.add(si)
                            if si in S_map:
                                new_j = set.union(new_j,S_map[si])
        S_set.append(sorted(temp_S))
        T_set.append(sorted(temp_T))
    return S_set,T_set

def write_senctence_pairs(S_set,T_set,C_file_name,E_file_name):
    C_sen_list,C_posi,C_len = C_process(C_file_name)
    E_sen_list,E_posi,E_len = E_process(E_file_name)
    with open('./test_map/new_mapped_sentence/'+C_file_name.strip('C_').strip(r"pubmed/"),'w',encoding='utf-8') as f:
        for k in range(len(S_set)):
            for i in S_set[k]:
                f.write(str(i) + ' ')
            f.write('←→')
            for j in T_set[k]:
                f.write(str(j) + ' ')
            f.write('\n')
        f.write('\n')
        for k in range(len(S_set)):
            for i in S_set[k]:
                f.write('C ' + str(i) + ' :' + C_sen_list[i]+'\n')
            for j in T_set[k]:
                f.write('E ' + str(j) + ' :' + E_sen_list[j]+'\n')
            f.write('===============================================\n')
def generate_pairs(S_set,T_set,C_file_name,E_file_name):
    C_sen_list,C_posi,C_len = C_process(C_file_name)
    E_sen_list,E_posi,E_len = E_process(E_file_name)
    with open('./version1_results/mapped_sentence/'+re.sub(r'C_abstract/','',C_file_name),'w',encoding='utf-8') as f:
        for k in range(len(S_set)):
            for i in S_set[k]:
                f.write(C_sen_list[i])
            f.write('|||')
            for j in range(len(T_set[k])):
                f.write(E_sen_list[T_set[k][j]])
                if j < len(T_set[k]) - 1 and E_sen_list[T_set[k][j]][-1] != ' ':
                    f.write(' ')
            f.write('\n') 
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(file)
    return L


def get_missing_number(L : list):
    return set(range(min(L),max(L)+1))-set(L)
def has_missing_number(L : list):
    if len(get_missing_number(L)) > 0:
        return True
    return False
def missing_between(L1,L2):
    assert max(L1) < max(L2)
    if max(L1) == max(L2)-1:
        return False
    return True
    
    
def missing(S_set,T_set):
    assert len(S_set) == len(T_set)
    miss = set()
    K = len(S_set)
    for i in range(K):
        if has_missing_number(S_set[i]) or has_missing_number(T_set[i]):
            miss.add(i)

    temp_S,temp_T = [],[]
    for i in range(K):
        if i not in miss:
            temp_S.append(S_set[i])
            temp_T.append(T_set[i])
    return temp_S,temp_T

need_to_map = []
with open('random_choice_doc_list.txt','r') as f:
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 1:
            need_to_map.append(line[0])
print(len(need_to_map))
for file in tqdm(need_to_map):
    try:
        C_file_name = 'C_abstract/'+file
        E_file_name = 'E_abstract/'+file
        all_inc2 = [];all_P = []
        z,A,N,M,C_len,E_len = get_z_A(C_file_name,E_file_name)
        if N <= 0 or M <= 0:
            continue
        all_epsilon = np.linspace(0,29,30)*0.01
        for epsilon in all_epsilon:
            P = with_slack(z,A,N,M,C_len,E_len,epsilon=epsilon)
            inc2 = sum_inconsistence2(P)
            all_inc2.append(inc2)
            all_P.append(P)

        e,P = get_epsilon(all_inc2,all_epsilon,all_P)
        P_clean = clean_P(P,N,M)
        S_map,T_map = get_mapping(P_clean,N,M)
        S_set,T_set = set_mapping(S_map,T_map)
        S_set,T_set = missing(S_set,T_set)
        
        generate_pairs(S_set,T_set,C_file_name,E_file_name)
    except:
        with open('new_generate_pairs_log.txt','a+',encoding='utf-8') as log_f:
            log_f.write(file+'\n')
            print('failed:',file)


