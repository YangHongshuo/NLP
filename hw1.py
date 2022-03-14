##hy2712

import argparse
import numpy as np
import features
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
##

def result_evaluation(topic_model, clf, x_test, y_test, y_predicted):
    print('LR Classification Report ' + topic_model)
    print(classification_report(y_test, y_predicted))

    print("Accuracy score: ", accuracy_score(y_test, y_predicted))

    # plot_confusion_matrix(clf, x_test, y_test)
    # plt.show()

##
def output_predictions(topic_model, y_predicted, output_path):
    # output_text = 'Prediction result for ' + topic_model + ': '
    output_text = ''
    for p in y_predicted:
        if p == 1:
            output_text += 'Pro\n'
        else:
            output_text += 'Con\n'
    #write to output file
    output_file = open(output_path, 'w')
    output_file.write(output_text)
    output_file.close()

##
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

    # get all features
    # x_train, y_train = features.get_features(args.train, args.user_data, args.model)
    # x_test, y_test = features.get_features(args.test, args.user_data, args.model)
    ##
    # USER_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/resources/data/users.json'
    # TRAIN_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/resources/data/train.jsonl'
    # TEST_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/resources/data/val.jsonl'
    # OUTPUT_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/output.txt'
    # LEX_PATH = '/Users/yhs/Columbia MSCS/2021 Fall/COMS W4705/HW1/Homework1/resources/lexica/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
    ##
    MODEL = 'Ngram+Lex'
    ##
    USER_PATH = args.user_data
    TRAIN_PATH = args.train
    TEST_PATH = args.test
    LEX_PATH = args.lexicon_path
    OUTPUT_PATH = args.outfile
    MODEL = args.model

    ##
    df_user = pd.read_json(USER_PATH, orient='index')
    df_train = pd.read_json(TRAIN_PATH, lines=True)
    df_test = pd.read_json(TEST_PATH, lines=True)

    df_train_religion = df_train[df_train['category'] == 'Religion']
    df_train_nonreligion = df_train[df_train['category'] != 'Religion']
    df_test_religion = df_test[df_test['category'] == 'Religion']
    df_test_nonreligion = df_test[df_test['category'] != 'Religion']
    ##
    index_religion = np.asarray(df_test_religion.index)
    index_nonreligion = np.asarray(df_test_nonreligion.index)
    index_test = np.concatenate((index_religion, index_nonreligion))

    ##
    x_train_religion, x_test_religion = features.get_features(df_train_religion, df_test_religion, df_user, MODEL, 'Religion', LEX_PATH)
    y_train_religion, y_test_religion = features.get_labels(df_train_religion, df_test_religion)
    x_train_nonreligion, x_test_nonreligion = features.get_features(df_train_nonreligion, df_test_nonreligion, df_user, MODEL, 'Non Religion', LEX_PATH)
    y_train_nonreligion, y_test_nonreligion = features.get_labels(df_train_nonreligion, df_test_nonreligion)

    ## scale all features before fitting the model
    scaler = MaxAbsScaler()
    scaler.fit_transform(x_train_religion)
    scaler.transform(x_test_religion)
    scaler.fit_transform(x_train_nonreligion)
    scaler.transform(x_test_nonreligion)
    ##
    # clf_religion = LogisticRegressionCV(
    #     penalty='none',
    #     cv=10,
    #     max_iter=600,
    # )
    ##
    if MODEL == 'Ngram' or MODEL == 'Ngram+Lex':
        clf_religion = LogisticRegression(
            C=21.5443469)
    else:
        clf_religion = LogisticRegression(
            C=2.7825594)
    clf_religion.fit(x_train_religion, y_train_religion)
    y_predicted_religion = clf_religion.predict(x_test_religion)
    result_evaluation('Religion (' + MODEL + ')', clf_religion, x_test_religion, y_test_religion, y_predicted_religion)
    ##
    clf_nonreligion = LogisticRegression()
    # clf_nonreligion = LogisticRegressionCV(
    #     cv=3,
    #     max_iter=500
    # )
    clf_nonreligion.fit(x_train_nonreligion, y_train_nonreligion)
    y_predicted_nonreligion = clf_nonreligion.predict(x_test_nonreligion)
    result_evaluation('Non Religion (' + MODEL + ')', clf_nonreligion, x_test_nonreligion, y_test_nonreligion, y_predicted_nonreligion)

    ## rearrange the results to match input order
    y_test = np.concatenate((y_test_religion, y_test_nonreligion))
    y_predicted = np.concatenate((y_predicted_religion, y_predicted_nonreligion))
    y_test = y_test[index_test]
    y_predicted = y_predicted[index_test]

    ##
    print("Balanced accuracy score: ", accuracy_score(y_test, y_predicted))
    ##
    output_predictions(MODEL, y_predicted, OUTPUT_PATH)