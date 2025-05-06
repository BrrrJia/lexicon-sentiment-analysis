import spacy
import pandas as pd


def read_instances(txt_file: str) -> list:
    """
    reads review corpus and lemmatizes it
    :param txt_file: the review txt file
    :return lemma_instances: a list of the lemmatized instances
    """
    with open(txt_file, 'r') as txt:
        raw_instances = txt.read().rstrip('\n').split('\n')
        nlp = spacy.load("de_core_news_sm")
        lemma_instances = []
        token_instances = []
        # save each instance as tuple of instance number and content of review
        instances = [(i[:4], i[5:]) for i in raw_instances]  # reviews start six characters into the rows
        for i in instances:
            instance_number = i[0]
            # tokenize the instance content
            tokens = nlp(i[1])
            token_instances.append(tokens)
            lemma_tokens = []
            for t in tokens:
                # lemmatize only the alphabetic tokens
                if t.is_alpha:
                    lemma_token = t.lemma_
                    lemma_tokens.append(lemma_token)        
            lemma_instances.append((instance_number, lemma_tokens))
    return lemma_instances


def read_lexicon(txt_file: str) -> dict:
    """
    read the lexicon file and make it in the form of dictionary
    :param txt_file: the review txt file
    :return lexicon_dict: lexicon in the form of dictionary
                            key: word
                            value: (tag, type)
    """
    with open(txt_file, 'r') as lexicon_file:
        lexicon = lexicon_file.read().split('\n')
        words = []
        features = []
        for elem in lexicon:
            if elem and not elem.startswith("%"):
                entry = elem.split()
                # separate word from values
                words.append(entry[0])
                features.append((entry[1],entry[2]))
        lexicon_dict = dict(zip(words, features))
    return lexicon_dict


def label_sentiment(lemmatized_review: list[str], lexicon: dict) -> str:
    """
    analyzes the sentiment of a lemmatized review and assigns it a label
    :param lemmatized_review: list of lemmata in a review
    :param lexicon: dict produced with read_lexicon(), containing words as
    keys and sentiment information as values
    :return: label: string indicating sentiment of the analyzed review
    """
    # default label is 'neutral' (no occurrences of positive or negative words)
    label = 'neutral'
    pos_words, neg_words = 0, 0  # track occurrences of positive or negative words
    for lemma in lemmatized_review:
        try:
            sentiment = lexicon[lemma][0]
            # check if lemma is positive or negative
            if sentiment[:3] == 'NEG':
                neg_words += 1
            elif sentiment[:3] == 'POS':
                pos_words += 1
        except:  # if lemma not in lexicon
            pass
    # label review according to count of positive and negative words
    if pos_words == 0 and neg_words > 0:
        label = 'negativ'
    elif neg_words == 0 and pos_words > 0:
        label = 'positiv'
    elif pos_words > 0 and neg_words > 0:
        label = 'gemischt'
    return label


def save_sentiments(list_of_reviews: list[tuple[str]], lexicon: dict) -> pd.DataFrame:
    """
    analyze all reviews in a corpus and save their sentiment label to dataframe
    :param list_of_reviews: list of lemmatized reviews, resulting from read_instances
    :param lexicon: lexicon for sentiment analysis (dict), resulting from read_lexicon
    :return: sentiment_table: pd.DataFrame listing each review and its sentiment
    """
    # intitialize dataframe for saving reviews and their sentiment labels
    sentiment_table = pd.DataFrame(columns=['review_nr', 'label'])
    for review in list_of_reviews:
        review_nr = review[0]
        lemma_list = review[1]
        review_label = label_sentiment(lemma_list, lexicon)
        new_row = pd.DataFrame([[review_nr, review_label]], columns=sentiment_table.columns)
        sentiment_table = pd.concat([sentiment_table, new_row], ignore_index=True)
    return sentiment_table


def combine_annotations(sent_table: pd.DataFrame, manual_path: str):
    """
    add labels from manual annotation into dataframe as new column
    :param sent_table: pd.DataFrame containing reviews and their labels
    :param manual_path: path where csv file with manual annotations is located
    :return: sent_table: modified pd.Dataframe with added colum for manual annotation
    """
    manual_annotation = pd.read_csv(manual_path)
    sent_table['manual_label'] = manual_annotation['Annotation']
    sent_table.to_csv('annotation_table.csv')
    return sent_table


def compare_annotations(df: pd.DataFrame, gold_column: str, model_column: str, labels: list[str]) -> dict:
    """
    counts true positives, false positives, false negatives for each label
    :param df: DataFrame containing reviews and their manually and automatically annotated sentiment labels
    :param gold_column: string indicating the name of the column considered the gold standard
    :param model_column: string indicating the name of the column considered the model to be compared to gold
    :param labels: list of label categories for which comparison is intended
    :return: label_dict: dict with labels as keys and tuple of true positives,
    false positives and false negatives (in that order) as values
    """
    label_dict = {}
    for label in labels:
        tp_subset = df[(df[model_column] == label) & (df[gold_column] == label)]
        true_positives = len(tp_subset)
        fp_subset = df[(df[model_column] == label) & (df[gold_column] != label)]
        false_positives = len(fp_subset)
        fn_subset = df[(df[gold_column] == label) & (df[model_column] != label)]
        false_negatives = len(fn_subset)
        label_dict[label] = (true_positives, false_positives, false_negatives)
    return label_dict


'''------------functions for evaluation scores------------'''


def get_precision(tp: int, fp: int) -> float:
    """
    calculate precision of a model for a category using given true positives and false positives
    :param tp: number of true positives
    :param fp: number of false positives
    :return: precision, None in case of division by zero
    """
    try:
        precision = tp/(tp+fp)
    except:
        precision = None
    return precision


def get_recall(tp: int, fn: int) -> float:
    """
        calculate recall of a model for a category using given true positives and false negatives
        :param tp: number of true positives
        :param fn: number of false negatives
        :return: recall, None in case of division by zero
    """
    try:
        recall = tp/(tp+fn)
    except:
        recall = None
    return recall


def get_f1_score(precision: float, recall: float) -> float:
    """
        calculate f1 score of a model for a category using given precision and recall values
        :param precision
        :param recall
        :return: f1 score, None in case of division by zero
       """
    try:
        if (precision and recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
    except:
        f1 = None
    return f1


if __name__ == "__main__":
    # process lexicon
    lexicon = read_lexicon("polartlexicon.txt")
    # read in corpus and lemmatize reviews
    lemmatized_reviews = read_instances('20_game_review.txt')
    sentiment_labels_model = save_sentiments(lemmatized_reviews, lexicon)
    data = combine_annotations(sentiment_labels_model, 'manual_annotation.csv')
    # compare the two annotations
    label_comparison = compare_annotations(
        data,
        'manual_label', 'label',
        ['positiv', 'negativ', 'neutral', 'gemischt'])
    # print results of precision, recall and f1 score calculations for each label
    for label, values in label_comparison.items():
        print(label + ': ')
        precision = get_precision(values[0], values[1])
        recall = get_recall(values[0], values[2])
        print('precision:', precision)
        print('recall:', recall)
        print('F1 score:', get_f1_score(precision, recall))
        print('\n')
