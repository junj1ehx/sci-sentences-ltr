import os
import nltk
from ext.rank_bm25 import BM25Okapi
import numpy as np
import shutil
import random


listPath = 'F:\PycharmProject\deim\one-column.txt'
output = 'F:\PycharmProject\deim\DocBank_500K_txt\ddataset'
docbank = 'F:\PycharmProject\deim\DocBank_500K_txt\DocBank_500K_txt'


def extract_file():
    with open(listPath, 'r', encoding='utf-8')as f:
        filelist = f.read().split('\n')
    for file in filelist:
        shutil.copyfile(os.path.join(docbank, file), os.path.join(output, file))

def openfile(filename_list_path):
    with open(filename_list_path, 'r', encoding='utf-8') as f:
        return [file.split("\n") for file in f.readlines()]

def BM25_as_relevence_score(caption, doc):
    # calculate BM25 as "fake" relevence score, return as list
    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(doc)

    tokenized_corpus = [doc.split(" ") for doc in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    query = caption
    tokenized_query = query.split(" ")
    cap_doc_scores = bm25.get_scores(tokenized_query)
    # print(cap_doc_scores)
    rel = np.zeros(len(cap_doc_scores))
    # for i in range(len(cap_doc_scores)):
    #     if (cap_doc_scores[i] <= 0):
    #         rel[i] = 0
    #     else:
    #         rel[i] = cap_doc_scores[i]
    # return rel

    for i in range(len(cap_doc_scores)):
        if (cap_doc_scores[i] <= 0):
            rel[i] = 0
        elif (cap_doc_scores[i] > 0 and cap_doc_scores[i] <= 1.508445):
            rel[i] = 1
        elif (cap_doc_scores[i] > 1.508445 and cap_doc_scores[i] <= 3.282216):
            rel[i] = 2
        elif (cap_doc_scores[i] > 3.282216 and cap_doc_scores[i] <= 6.384559):
            rel[i] = 3
        elif (cap_doc_scores[i] > 6.384559 and cap_doc_scores[i] <= 12.453971):
            rel[i] = 4
        elif (cap_doc_scores[i] > 12.453971):
            rel[i] = 5
        else:
            rel[i] = 0
    return rel

    # rel = np.zeros(len(cap_doc_scores), dtype = np.int)
    # for i in range(len(cap_doc_scores)):
    #     if (cap_doc_scores[i] <= 0):
    #         rel[i] = 0
    #     else:
    #         rel[i] = 1
    #
    # return rel

def openfile(filename_list_path):
    with open(filename_list_path, 'r', encoding='utf-8') as f:
        return [file.split("\n") for file in f.readlines()]

def feature_extraction(table, doc):
    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(doc)
    tokenized_corpus = [doc.split(" ") for doc in sentences]
    doc_bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = table.split(" ")
    tokenized_query.pop() # remove last empty word

    doc_num, doc_nonnum = doc_number_count(tokenized_corpus)

    table_doc_scores = doc_bm25.get_scores_extended(tokenized_query)
    for i in range(len(tokenized_corpus)):
        table_doc_scores[i][8] = doc_num[i]
        table_doc_scores[i][9] = doc_nonnum[i]
    # print(table_doc_scores)
    return table_doc_scores

def doc_number_count(query):
    # count number of numbers in document
    doc_num = np.zeros(len(query))
    doc_nonnum = np.zeros(len(query))

    for i in range(len(query)):
        tagged_tokenized_query = nltk.pos_tag(query[i])
        count_num = 0
        for j in range(len(query[i])):
            if (tagged_tokenized_query[j][1] == 'CD'):
                count_num += 1
        doc_num[i] = count_num / len(query) # ratio of number of D
        doc_nonnum[i] = (len(query) - count_num) / len(query)  # ratio of non number of D
    return doc_num, doc_nonnum


def preprocess(file_name, docid):
    # with open('handcrafted/' + file_name, 'r', encoding='utf-8') as f:
    with open('extracted/' + file_name, 'r', encoding='utf-8') as f:
        raw = [i[:-1].split('\t') for i in f.readlines()]

    paragraph = ''
    table = ''
    caption = ''

    # generate 3 kinds of input
    table_raw_all = []
    table_all = ''
    # table_column = ''
    table_row = ''
    table_others = ''
    count_table = 0
    for i in range(len(raw)):
        if (raw[i][0] != '##LTLine##' and raw[i][9] == 'table'):
            table_raw_all.append(raw[i])
            table_all = table_all + raw[i][0] + ' '
            count_table += 1

    flag = 0  # 0 row 2 others
    for i in range(count_table):
        if i == 0 and flag == 0:
            # if first come
            table_row_y0 = table_raw_all[i][2]
            table_row = table_row + table_raw_all[i][0] + ' '
        elif i > 0 and table_row_y0 == table_raw_all[i][2] and flag == 0:
            # if still in the same line
            table_row_y0 = table_raw_all[i][2]
            table_row = table_row + table_raw_all[i][0] + ' '
        elif i > 0 and table_row_y0 != table_raw_all[i][2] and flag == 0:
            # if line change are detected
            table_row_y0 = table_raw_all[i][2]
            table_others = table_others + table_raw_all[i][0] + ' '
            flag = 1
        else:
            table_others = table_others + table_raw_all[i][0] + ' '

    for i in range(len(raw)):
        if (raw[i][0] != '##LTLine##' and raw[i][9] == 'paragraph'):
            paragraph = paragraph + raw[i][0] + ' '
        if (raw[i][9] == 'caption'):
            caption = caption + raw[i][0] + ' '
        if (raw[i][0] != '##LTLine##' and raw[i][9] == 'table'):
            table = table + raw[i][0] + ' '

    rel = BM25_as_relevence_score(caption, paragraph)
    score_all = feature_extraction(table_all, paragraph)
    # score_column = feature_extraction(table_column, paragraph)
    score_row = feature_extraction(table_row, paragraph)
    score_oth = feature_extraction(table_others, paragraph)

    qid = docid # temp qid
    doc_id_count = 1 # temp doc_id_count
    with open('output.txt', 'a+', encoding = 'utf-8') as f:
        for i in range(len(rel)):
            output = str(rel[i]) + ' qid:' + str(qid) + ' '
            for j in range(10):
                output = output + str(1 + j*4) + ':' + str('%.6f' % score_all[i][j]) + ' ' + str(2 + j*4) + ':' + str('%.6f' % score_column[i][j]) + ' ' + str(3 + j*4) + ':' + str('%.6f' % score_row[i][j]) + ' ' + str(4 + j*4) + ':' + str('%.6f' % score_oth[i][j]) + ' '
            output = output + '#docid = ' + str(qid) + '-' + str(doc_id_count)
            doc_id_count += 1
            f.write(output + '\n')

if __name__ == '__main__':
    filename_list_path = 'filelist.txt'
    filename_list = openfile(filename_list_path)
    print("extracting features from documents...")

    for i in range(len(filename_list)):
        preprocess(filename_list[i][0], i+1)
        with open('log.txt', 'a+', encoding='utf-8') as f:
            f.write(filename_list[i][0] + '\n')
        print('file' + filename_list[i][0] + ' preprocessed' + '(' + str(i+1) + '/' + str(1000) + ')')

    print("dividing files...")
    row_count = 0
    with open('output.txt', 'r') as f:
        ori = []
        for line in f:
            ori.append(line)
            row_count += 1
        print(str(row_count) + 'query-document pairs in total')
        print(ori)
        divide_num = row_count / 5
        for i in range(len(ori)):
            temp_num = random.randint(1,5)
            with open('dataset/' + 'S' + str(temp_num) + '.txt', 'a+', encoding='utf-8') as file:
                file.write(ori[i])
                print(str(i + 1) + '/' + str(len(ori)) + '\n')