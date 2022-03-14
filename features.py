##hy2712

import numpy as np
import pandas as pd
# import scipy
from nltk import word_tokenize
from scipy.sparse import coo_matrix, hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from os import path
from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_selection import SelectKBest, chi2

##
def get_labels(df_train, df_test):
    y_train = get_y(df_train)
    y_test = get_y(df_test)
    return y_train, y_test

##
def get_features(df_train, df_test, df_user, model, topic, lex_path):
    # stack different features based on model selections
    x_train, x_test = np.empty((df_train.shape[0],0)), np.empty((df_test.shape[0],0))

    ## get all texts from debate rounds
    texts_train_pro = get_debate_text(df_train, 'Pro')
    texts_train_con = get_debate_text(df_train, 'Con')
    texts_test_pro = get_debate_text(df_test, 'Pro')
    texts_test_con = get_debate_text(df_test, 'Con')
    debate_texts = (texts_train_pro, texts_train_con, texts_test_pro, texts_test_con)
    ##
    if 'Ngram' in model:
        #
        ngram_train_path = 'ngram_train_matrix_' + topic + '.npz'
        ngram_test_path = 'ngram_test_matrix_' + topic + '.npz'
        #
        if path.exists(ngram_train_path) and path.exists(ngram_train_path):
            x_train = load_npz(ngram_train_path)
            x_test = load_npz(ngram_test_path)
        else:
            ## get n_grams
            n_grams_train_pro, n_grams_test_pro, n_grams_train_con, n_grams_test_con = get_ngrams(debate_texts, topic)
            x_train = hstack([x_train, n_grams_train_pro])
            x_train = hstack([x_train, n_grams_train_con])
            x_test = hstack([x_test, n_grams_test_pro])
            x_test = hstack([x_test, n_grams_test_con])
            save_npz(ngram_train_path, x_train)
            save_npz(ngram_test_path, x_test)

    ##
    if 'Lex' in model:
        lex_train_path = 'lex_train_matrix_' + topic + '.npz'
        lex_test_path = 'lex_test_matrix_' + topic + '.npz'
        if path.exists(lex_train_path) and path.exists(lex_test_path):
            lex_scores_train = load_npz(lex_train_path)
            lex_scores_test = load_npz(lex_test_path)
        else:
            # get lexicons
            LEX_DICT = 'VAD'
            lex_scores_train = get_lex(texts_train_pro, texts_train_con, LEX_DICT, lex_path)
            lex_scores_test = get_lex(texts_test_pro, texts_test_con, LEX_DICT, lex_path)
            save_npz(lex_train_path, lex_scores_train)
            save_npz(lex_test_path, lex_scores_test)
        x_train = hstack([x_train, lex_scores_train])
        x_test = hstack([x_test, lex_scores_test])
    ##
    if 'Ling' in model:
        ling_train_path = 'ling_train_matrix_' + topic + '.npz'
        ling_test_path = 'ling_test_matrix_' + topic + '.npz'
        if path.exists(ling_train_path) and path.exists(ling_test_path):
            ling_train = load_npz(ling_train_path)
            ling_test = load_npz(ling_test_path)
        else:
            # get linguistic features
            ling1_train, ling1_test, ling2_train, ling2_test = get_ling(debate_texts)
            ling_train = hstack([ling1_train, ling2_train])
            ling_test = hstack([ling1_test, ling2_test])
            save_npz(ling_train_path, ling_train)
            save_npz(ling_test_path, ling_test)
        x_train = hstack([x_train, ling_train])
        x_test = hstack([x_test, ling_test])

    if 'User' in model:
        # get user features
        # pro_train_features, pro_test_features, con_train_features, con_test_features = get_user_features(df_train, df_test, df_user, topic)
        # x_train = hstack([x_train, pro_train_features])
        # x_train = hstack([x_train, con_train_features])
        # x_test = hstack([x_test, pro_test_features])
        # x_test = hstack([x_test, con_test_features])

        # ideo_train, ideo_test = get_user_features2(df_train, df_test, df_user)
        # x_train = hstack([x_train, ideo_train])
        # x_test = hstack([x_test, ideo_test])
        user_train_path = 'user_train_matrix_' + topic + '.npz'
        user_test_path = 'user_test_matrix_' + topic + '.npz'
        if path.exists(user_train_path) and path.exists(user_test_path):
            user_train = load_npz(user_train_path)
            user_test = load_npz(user_test_path)
        else:
            reli_ideo_train, poli_ideo_train, reli_ideo_test, poli_ideo_test = get_user_features(df_train, df_test, df_user)
            user_train = hstack([reli_ideo_train, poli_ideo_train])
            user_test = hstack([reli_ideo_test, poli_ideo_test])
            save_npz(user_train_path, user_train)
            save_npz(user_test_path, user_test)
        x_train = hstack([x_train, user_train])
        x_test = hstack([x_test, user_test])

    return x_train, x_test

##
def get_ngrams(debate_texts, topic = 'Religion'):
     ## get n_grams using vectorizer
    if topic == 'Religion':
        vectorsizer = TfidfVectorizer(
            ngram_range=(1, 3),
            tokenizer=word_tokenize,
            max_df=0.85,
            min_df=0.15,
            # max_features=8000
        )
    else:
        vectorsizer = TfidfVectorizer(
            ngram_range=(1, 3),
            tokenizer=word_tokenize,
            max_df=0.90,
            min_df=0.10,
            max_features=8000
        )
    n_grams_train_pro = vectorsizer.fit_transform(debate_texts[0])
    n_grams_test_pro = vectorsizer.transform(debate_texts[2])
    n_grams_train_con = vectorsizer.fit_transform(debate_texts[1])
    n_grams_test_con = vectorsizer.transform(debate_texts[3])
##
    return n_grams_train_pro, n_grams_test_pro, n_grams_train_con, n_grams_test_con

##
def get_debate_text(df_debate, side):
    texts = []
    df_rounds = df_debate['rounds']
    for row in range(df_rounds.shape[0]):
        text = get_debate_text2(df_rounds.iloc[row], side)
        texts.append(text)
    return texts

##
def get_debate_text2(round, side):
    text = ''
    for i in range(len(round)):
        for j in range(len(round[i])):
            if round[i][j]['side'] == side:
                text += round[i][j]['text']
    return text

##
def get_lex(pro_texts, con_texts, lex_dict = 'VAD', lex_path = ''):
    scores = []
    ##
    # lex_dict='CONN'
    for row in range(len(pro_texts)):
        pro_text = pro_texts[row]
        con_text = con_texts[row]

        if lex_dict == 'VAD':
            ## Load the Lexicon to dataframe
            # LEX_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/resources/lexica/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
            LEX_PATH = lex_path
            df_lex = pd.read_csv(LEX_PATH, sep="	", header=None, names=['V', 'A', 'D'])
            dict_lex = df_lex.to_dict('index')
            ## VAD scores
            # pro_v_score, pro_a_score, pro_d_score = get_lex_VAD_scores(df_lex, pro_text)
            # con_v_score, con_a_score, con_d_score = get_lex_VAD_scores(df_lex, con_text)
            pro_v_score, pro_a_score, pro_d_score = get_lex_VAD_scores2(dict_lex, pro_text)
            con_v_score, con_a_score, con_d_score = get_lex_VAD_scores2(dict_lex, con_text)
            ##
            scores.append([pro_v_score, pro_a_score, pro_d_score, con_v_score, con_a_score, con_d_score])
    ##
    score_matrix = coo_matrix(scores)
    return score_matrix

##
def get_lex_VAD_scores(df_lex, text):
    # get tokens from the text
    # text = df_debate['rounds'][0][0][0]['text']
    # type = 'V'
    tokens = word_tokenize(text)
    ## Get scroes based on lexicon dict
    v_score, a_score, d_score = 0, 0, 0
    word_count = 0
    for word in tokens:
        if word in df_lex.index:
            v_score += df_lex.loc[word, 'V']
            a_score += df_lex.loc[word, 'A']
            d_score += df_lex.loc[word, 'D']
            word_count += 1
    ##
    if word_count == 0:
        print('count 0')
        word_count = 1
    ##
    return v_score/word_count, a_score/word_count, d_score/word_count

##
def get_lex_VAD_scores2(dict_lex, text):
    # get tokens from the text
    # text = df_debate['round'][0][0][0]['text']
    tokens = word_tokenize(text)

    # Get scroes based on lexicon dict
    v_score, a_score, d_score = 0, 0, 0
    word_count = 0
    for word in tokens:
        if word in dict_lex:
            v_score += dict_lex[word]['V']
            a_score += dict_lex[word]['A']
            d_score += dict_lex[word]['D']
            word_count += 1
    #
    if word_count == 0:
        word_count = 1
    #
    return v_score/word_count, a_score/word_count, d_score/word_count

##
def get_ling(debate_texts):
    exclamation_train = get_exclamation_counts(debate_texts[0], debate_texts[1])
    exclamation_test = get_exclamation_counts(debate_texts[2], debate_texts[3])
    length_train = get_length(debate_texts[0], debate_texts[1])
    length_test = get_length(debate_texts[2], debate_texts[3])
    return exclamation_train, exclamation_test, length_train, length_test

##
def get_exclamation_counts(pro_texts, con_texts):
    ##
    pro_exclamation = []
    con_exclamation = []
    for row in range(len(pro_texts)):
        pro_exclamation.append(pro_texts[row].count('!'))
        con_exclamation.append(con_texts[row].count('!'))
    ##
    exclamation_counts = np.transpose([pro_exclamation, con_exclamation])
    exclamation_counts_matrix = coo_matrix(exclamation_counts)
    ##
    return exclamation_counts_matrix

##
def get_length(pro_texts, con_texts):
    ##
    text_length = []
    for row in range(len(pro_texts)):
        # pro_tokens = word_tokenize(pro_texts[row])
        # con_tokens = word_tokenize(con_texts[row])
        # pro_count = len(pro_tokens)
        # con_count = len(con_tokens)
        pro_count = len(pro_texts[row])
        con_count = len(con_texts[row])
        total_count = pro_count + con_count
        if total_count == 0:
            text_length.append([0, 0])
        else:
            text_length.append([pro_count/total_count, con_count/total_count])

    length_matrix = coo_matrix(text_length)
    return length_matrix

##
def get_user_features(df_train, df_test, df_user):
    ## map useful user features to the debate table
    df_user_ideo = df_user.loc[:, ['religious_ideology', 'political_ideology']]
    df_train_ideo = df_train.join(df_user_ideo, on='pro_debater')
    df_train_ideo = df_train_ideo.join(df_user_ideo, on='con_debater', lsuffix='_pro', rsuffix='_con')
    df_test_ideo = df_test.join(df_user_ideo, on='pro_debater')
    df_test_ideo = df_test_ideo.join(df_user_ideo, on='con_debater', lsuffix='_pro', rsuffix='_con')

    ##
    reli_ideo_scores_train, reli_ideo_scores_test = [], []
    poli_ideo_scores_train, poli_ideo_scores_test = [], []
    for index, row in df_train_ideo.iterrows():
        reli_ideo_scores_train.append(check_match_ideo(row, df_user_ideo, 'religious'))
        poli_ideo_scores_train.append(check_match_ideo(row, df_user_ideo, 'political'))
    for index, row in df_test_ideo.iterrows():
        reli_ideo_scores_test.append(check_match_ideo(row, df_user_ideo, 'religious'))
        poli_ideo_scores_test.append(check_match_ideo(row, df_user_ideo, 'political'))

    reli_matrix_train = coo_matrix(reli_ideo_scores_train)
    poli_matrix_train = coo_matrix(poli_ideo_scores_train)
    reli_matrix_test = coo_matrix(reli_ideo_scores_test)
    poli_matrix_test = coo_matrix(poli_ideo_scores_test)
##
    return reli_matrix_train, poli_matrix_train, reli_matrix_test, poli_matrix_test

##
def check_match_ideo(row, df_user, match_type = 'religious'):
    pro_ideo_score = 0
    con_ideo_score = 0
    # row = df_train_ideo.iloc[10]
    # match_type = 'religious'
    # print(row['voters'])
    for voter in row['voters']:
        if voter in df_user.index:
            voter_ideo = df_user.loc[voter, match_type + '_ideology']
            if voter_ideo == row[match_type + '_ideology_pro']:
                pro_ideo_score += 1
            if voter_ideo == row[match_type + '_ideology_con']:
                con_ideo_score += 1

    total_voters = len(row['voters'])
    if total_voters == 0:
        return 0,0
    else:
        return [pro_ideo_score/total_voters, con_ideo_score/total_voters]

##
def get_opinion_similarity(df_train, df_test, df_user):
    similarity_train, similarity_test = [], []
    for index, row in df_train.iterrows():
        scores = get_similarity_score(row, df_user)
        similarity_train.append(scores)
    for index, row in df_test.iterrows():
        scores = get_similarity_score(row, df_user)
        similarity_test.append(scores)
    return coo_matrix(similarity_train), coo_matrix(similarity_test)

##
def get_similarity_score(row, df_user):
    pro_dict = df_user.loc[row['pro_debater'], 'big_issues_dict']
    con_dict = df_user.loc[row['con_debater'], 'big_issues_dict']
    return 0

##
def big_issue_dic_to_array(issues_dict):
    dict_num = []
    for key, value in issues_dict.items():
        if value == 'N/S':
            dict_num.append(0)
        else:
            dict_num.append(1)
    return np.asarray(dict_num)

##
def get_y(df_debate):
    y_values = np.empty((df_debate.shape[0]),dtype='int')
    df_winner = df_debate['winner']
    for i in range(df_winner.shape[0]):
        if df_winner.iloc[i] == 'Pro':
            y_values[i] = 1
        else:
            y_values[i] = 0
    return y_values

