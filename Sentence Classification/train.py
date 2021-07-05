import os
import string
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_hub as hub
import  matplotlib.pyplot as plt
from spacy.lang.en import English
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# import option function
from preprocess_func import preprocess_text_with_line_numbers, split_chars
from pred_func import get_pred


# Load and Preprocessing Data
## Load Data
data_dir = "G:/Python/SCMPA/data/PubMed_20k_RCT_numbers_replaced_with_at_sign"
#data_dir = "data/PubMed_20k_RCT_numbers_replaced_with_at_sign"
filenames = [data_dir + filename for filename in os.listdir(data_dir)]

## get data samples
train_samples = preprocess_text_with_line_numbers(data_dir + "/train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "/dev.txt") # dev is another name for validation set
test_samples = preprocess_text_with_line_numbers(data_dir + "/test.txt")
# print(len(train_samples))

##v Create train_df
train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)

## Convert abstract text lines into lists
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()

# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# Use TensorFlow to create one-hot-encoded tensors of our "line_number" column
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

# Use TensorFlow to create one-hot-encoded tensors of our "total_lines" column
train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)

# Create training and validation datasets (all four kinds of inputs)
train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot, # line numbers
                                                                train_total_lines_one_hot, # total lines
                                                                train_sentences, # train tokens
                                                                train_chars)) # train chars
train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # train labels
train_pos_char_token_dataset = tf.data.Dataset.zip((train_pos_char_token_data, train_pos_char_token_labels)) # combine data and labels
train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE) # turn into batches and prefetch appropriately

# Validation dataset
val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                              val_total_lines_one_hot,
                                                              val_sentences,
                                                              val_chars))
val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE) # turn into batches and prefetch appropriately


char_lens = [len(sentence) for sentence in train_sentences]
output_seq_char_len = int(np.percentile(char_lens, 95))

##
alphabet = string.ascii_lowercase + string.digits + string.punctuation
# Create char-level token vectorizer instance
NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")

# Create char embedding layer
char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS, # number of different characters
                              output_dim=25, # embedding dimension of each character (same as Figure 1 in https://arxiv.org/pdf/1612.05251.pdf)
                              mask_zero=True,
                              name="char_embed")




# Create a model
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

def get_model():
    # 1. Token inputs
    token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
    token_embeddings = tf_hub_embedding_layer(token_inputs)
    token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
    token_model = tf.keras.Model(inputs=token_inputs,
                                outputs=token_embeddings)

    # 2. Char inputs
    char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(32))(char_embeddings)
    char_model = tf.keras.Model(inputs=char_inputs,
                                outputs=char_bi_lstm)

    # 3. Line numbers inputs
    line_number_inputs = layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
    x = layers.Dense(32, activation="relu")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs=line_number_inputs,
                                        outputs=x)

    # 4. Total lines inputs
    total_lines_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
    y = layers.Dense(32, activation="relu")(total_lines_inputs)
    total_line_model = tf.keras.Model(inputs=total_lines_inputs,
                                        outputs=y)

    # 5. Combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([token_model.output,
                                                                              char_model.output])
    z = layers.Dense(256, activation="relu")(combined_embeddings)
    z = layers.Dropout(0.5)(z)

    # 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
    z = layers.Concatenate(name="token_char_positional_embedding")([line_number_model.output,
                                                                total_line_model.output,
                                                                z])

    # 7. Create output layer
    output_layer = layers.Dense(5, activation="softmax", name="output_layer")(z)

    # 8. Put together model
    model = tf.keras.Model(inputs=[line_number_model.input,
                                 total_line_model.input,
                                 token_model.input,
                                 char_model.input],
                                 outputs=output_layer)


    return model

## Train Model
model = get_model()

# Compile token, char, positional embedding model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), # add label smoothing (examples which are really confident get smoothed a little)
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
## check point
# checkpoint = ModelCheckpoint('G:/Python/SCMPA/models/jsmodel/best.hdf5',
#                                  monitor = 'val_loss',
#                                  save_best_only=True, mode='auto')
# callback_list = [checkpoint]
# Fit the token, char and positional embedding model
history_model = model.fit(train_pos_char_token_dataset,
                              steps_per_epoch=int(0.1 * len(train_pos_char_token_dataset)),
                              epochs=3,
                              validation_data=val_pos_char_token_dataset,
                              validation_steps=int(0.1 * len(val_pos_char_token_dataset)))
                              #callbacks=callback_list)

## Save models
#model.save_weights('G:/Python/SCMPA/models/model.hdf5')

# Create test dataset batch and prefetched
test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                               test_total_lines_one_hot,
                                                               test_sentences,
                                                               test_chars))
test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels))
test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Confusion matrix
# Extract labels ("target" columns) and encode them into integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

# Make predictions on the test dataset
test_pred_probs = model.predict(test_pos_char_token_dataset,
                                       verbose=1)
test_preds = tf.argmax(test_pred_probs, axis=1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cfn_matrix = confusion_matrix(test_labels_encoded, test_preds)
print(classification_report (test_labels_encoded, test_preds))
print(cfn_matrix)

label = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS' ]
df_cfn  = pd.DataFrame(cfn_matrix, range(5), range(5))
plt.figure(figsize = (10,10))
sn.set(font_scale = 1.4 )
sn.heatmap(df_cfn, annot = True, annot_kws={"size": 16})
tick_marks = np.arange(len(label))
plt.xticks(tick_marks, label, rotation =45)
plt.yticks(tick_marks, label, rotation = -45)
plt.show()

## Predict
def get_pred(filename):
    result_file_path = 'G:/Python/SCMPA/data/result_pred/result_' + filename
    filename = 'G:/Python/SCMPA/data/pred_data/' + filename

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

    f = open(result_file_path, 'w')
    for i, line in enumerate(abstract_lines):
        print(f"{test_abstract_pred_classes[i]}: {line}")
        f.writelines(f"{test_abstract_pred_classes[i]}: {line}\n")


txt_pred1 = 'pred1.txt'
#txt_pred1 = 'SCMPA/data/pred_data/pred1.txt'
get_pred(txt_pred1)

txt_pred2 = 'pred2.txt'
#txt_pred2 = 'data/pred_data/pred2.txt'
get_pred(txt_pred2)
