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
#cnt
    # for i in range(len(cap_doc_scores)):
    #     if cap_doc_scores[i] <= 0:
    #         rel[i] = 0
    #     else:
    #         rel[i] = cap_doc_scores[i]
    # return rel
#grd
    for i in range(len(cap_doc_scores)):
        if (cap_doc_scores[i] <= 0):
            rel[i] = 0
        elif (cap_doc_scores[i] > 0 and cap_doc_scores[i] <= 0.918817199):
            rel[i] = 1
        elif (cap_doc_scores[i] > 0.918817199 and cap_doc_scores[i] <= 1.75987574369279):
            rel[i] = 2
        elif (cap_doc_scores[i] > 1.75987574369279 and cap_doc_scores[i] <= 4.595513338):
            rel[i] = 3
        elif (cap_doc_scores[i] > 4.595513338 and cap_doc_scores[i] <= 7.76186500588613):
            rel[i] = 4
        elif (cap_doc_scores[i] > 7.76186500588613):
            rel[i] = 5
        else:
            rel[i] = 0
    return rel
# bin
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
    with open('deimDataset/ddataset/' + file_name, 'r', encoding='utf-8') as f:
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

    # output the raw
    # with open ('qd_raw.txt', 'a+', encoding = 'utf-8') as f:
    #     f.write('##########' + '\n' + '#qid:' + str(qid) + '\n' + '#rel:' + str(rel) + '\n' +
    #             '#caption:' + caption + '\n' +
    #             '#paragraph:' + paragraph + '\n' +
    #             '#table:' + table_all + '\n' +
    #             '#row:' + table_row + '\n' +
    #             '#other:' + table_others + '\n' + '##########' + '\n')

    # output the l2r
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    with open('dataset/output.txt', 'a+', encoding ='utf-8') as f:
        for i in range(len(rel)):
            output = str(rel[i]) + ' qid:' + str(qid) + ' '
            temp = []
            # 1 -> 4 12 -> 6 123 -> 10
            for j in range(10):
                if j < 4 or j > 5:
                #if j < 4:
                #if j > 5:
                    temp.append('%.6f' % score_all[i][j])
                    if j != 4 and j != 8 and j != 9:
                        temp.append('%.6f' % score_row[i][j])
                        temp.append('%.6f' % score_oth[i][j])
# note from cal
# score_metric[i][0] = total_q[i]
# score_metric[i][1] = total_idf[i]
# score_metric[i][2] = total_q[i] * total_idf[i]
# score_metric[i][3] = score[i]
# score_metric[i][4] = doc_len[i]
# score_metric[i][5] = query_len[i]
# score_metric[i][6] = query_num[i]
# score_metric[i][7] = query_nonnum[i]
# table_doc_scores[i][8] = doc_num[i]
# table_doc_scores[i][9] = doc_nonnum[i]


            # for i in range(len(temp)):
                # output = output + str('%.6f' % score_all[i][j]) + ' ' + str('%.6f' % score_row[i][j]) + ' ' + str('%.6f' % score_oth[i][j]) + ' '
            for i in range(len(temp)):
                output = output + str(1 + i) + ':' + temp[i] + ' '

            # (i) TF, IDF, TF*IDF, BM25
            # for j in range(4):
            #     output = output + str(1 + j*3) + ':' + str('%.6f' % score_all[i][j]) + ' ' + str(2 + j*3) + ':' + str('%.6f' % score_row[i][j]) + ' ' + str(3 + j*3) + ':' + str('%.6f' % score_oth[i][j]) + ' '
            # # (ii)
            # output = output + str(13) + ':' + str('%.6f' % score_all[i][4])
            # output = output + str(14) + ':' + str('%.6f' % score_all[i][5]) + ' ' + str(15) + ':' + str('%.6f' % score_row[i][5]) + ' ' + str(16) + ':' + str('%.6f' % score_oth[i][5]) + ' '
            # # (iii)
            # output = output + str(17) + ':' + str('%.6f' % score_all[i][6]) + ' ' + str(18) + ':' + str('%.6f' % score_row[i][6]) + ' ' + str(19) + ':' + str('%.6f' % score_oth[i][6]) + ' '
            # output = output + str(20) + ':' + str('%.6f' % score_all[i][7]) + ' ' + str(21) + ':' + str('%.6f' % score_row[i][7]) + ' ' + str(22) + ':' + str('%.6f' % score_oth[i][7]) + ' '
            # output = output + str(23) + ':' + str('%.6f' % score_all[i][8]) + ' '
            # output = output + str(24) + ':' + str('%.6f' % score_all[i][9]) + ' '
            output = output + '#docid = ' + str(qid) + '-' + str(doc_id_count)
            doc_id_count += 1
            f.write(output + '\n')

if __name__ == '__main__':
    filename_list_path = 'F:\PycharmProject\deim\deimDataset\one-column.txt'
    filename_list = openfile(filename_list_path)
    print("extracting features from documents...")

    for i in range(len(filename_list)):
        preprocess(filename_list[i][0], i+1)
        with open('old/log.txt', 'a+', encoding='utf-8') as f:
            f.write(filename_list[i][0] + '\n')
        print('file' + filename_list[i][0] + ' preprocessed' + '(' + str(i+1) + '/' + str(500) + ')')

    print("dividing files...")
    row_count = 0
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    with open('dataset/output.txt', 'r') as f:
        ori = []
        for line in f:
            ori.append(line)
            row_count += 1
        print(str(row_count) + 'query-document pairs in total')
        # print(ori)
        divide_num = row_count / 5
        for i in range(len(ori)):
            temp_num = random.randint(1,5)
            with open('dataset/' + 'S' + str(temp_num) + '.txt', 'a+', encoding='utf-8') as file:
                file.write(ori[i])
                print(str(i + 1) + '/' + str(len(ori)) + '\n')

    print("folding...")
    for i in range(5):
        path = 'dataset/Fold' + str(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/' + 'train.txt', 'a+', encoding='utf-8') as fw:
            for j in range(3):
                with open('dataset/S' + str((i + j) % 5 + 1) + '.txt', 'r', encoding='utf-8') as fr:
                    fw.write(fr.read())
        with open(path + '/' + 'vali.txt', 'a+', encoding='utf-8') as fw:
            with open('dataset/S' + str((i + 4) % 5 + 1) + '.txt', 'r', encoding='utf-8') as fr:
                fw.write(fr.read())
        with open(path + '/' + 'test.txt', 'a+', encoding='utf-8') as fw:
            with open('dataset/S' + str((i + 5) % 5 + 1) + '.txt', 'r', encoding='utf-8') as fr:
                fw.write(fr.read())