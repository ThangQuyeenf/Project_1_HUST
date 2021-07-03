import os
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from preprocess_func import split_chars


def get_pred(filename, model):
    with open(filename, 'r') as f:
        example_abstracts = f.read()

    example_abstracts = example_abstracts.split( sep = '.')
    abstract_lines = []
    for sentence in example_abstracts:
        sentence += '.'
        abstract_lines.append(sentence)
    abstract_lines.remove('.')


    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)


    # Get all line_number values from sample abstract
    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)


    # Get all total_lines values from sample abstract
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]


    # Make predictions on sample abstract features

    test_abstract_pred_probs = model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))


    # Extract labels ("target" columns) and encode them into integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
    val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
    test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]

    # Visualize abstract lines and predicted sequence labels
    for i, line in enumerate(abstract_lines):
        print(f"{test_abstract_pred_classes[i]}: {line}")
