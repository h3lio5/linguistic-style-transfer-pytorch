import re
import logging

train_pos_reviews_file_path = "data/raw/yelp_train_pos.txt"
train_neg_reviews_file_path = "data/raw/yelp_train_neg.txt"

train_text_file_path = "data/clean/yelp_train_data.txt"
train_labels_file_path = "data/clean/yelp_train_labels.txt"


def clean_text(string):
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\\n", " ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r'\d+', "number", string)
    string = string.replace("\r", " ")
    string = string.replace("\n", " ")
    string = string.strip().lower()

    return string


logging.info("Writing train dataset")
with open(train_text_file_path, 'w') as text_file, open(train_labels_file_path, 'w') as labels_file:
    with open(train_pos_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("pos" + "\n")
    with open(train_neg_reviews_file_path, 'r') as reviews_file:
        for line in reviews_file:
            text_file.write(clean_text(line) + "\n")
            labels_file.write("neg" + "\n")

logging.info("Processing complete")
