import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

reuse_model=True
train_model=True
model_name="model_simple_save/"
string_max_length=8
batch_size = 32
epochs = 2
HIDDEN_DIM = 512

train_data = pd.read_csv("fr_en.tsv", sep="\t", header=None, names = ['id1', 'fr', 'id2','en'])[['fr', 'en']]
train_data = train_data[train_data['fr'].str.contains('Je suis')]
train_data['fr']='<BOS> ' + train_data['fr'] + ' <EOS>'
train_data['en']='<BOS> ' + train_data['en'] + ' <EOS>'

num_words=15000
tokenizerFr = Tokenizer(num_words=num_words, lower=True, oov_token="<rare>")
tokenizerFr.fit_on_texts(train_data['fr'])
vocab_size_fr = min(len(tokenizerFr.word_index) + 1, num_words)

tokenizerEn = Tokenizer(num_words=num_words, lower=True, oov_token="<rare>")
tokenizerEn.fit_on_texts(train_data['en'])
vocab_size_en = min(len(tokenizerEn.word_index) + 1, num_words)

print(vocab_size_fr, vocab_size_en)

x_train, x_test, y_train, y_test =  train_test_split(train_data['fr'], train_data['en'])

class MySequence(tf.keras.utils.Sequence):
    def __init__(self, x_string, y_string, batch_size):
        self.x_string, self.y_string = x_string, y_string
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x_string) // self.batch_size

    def __getitem__(self, idx):
        seqFr = tokenizerFr.texts_to_sequences(self.x_string[idx * self.batch_size:(idx+1) * self.batch_size])
        seqFr = pad_sequences(seqFr, maxlen=string_max_length, truncating='post')
        seqEn = tokenizerEn.texts_to_sequences(self.y_string[idx * self.batch_size:(idx+1) * self.batch_size])
        seqEn = pad_sequences(seqEn, maxlen=string_max_length,  truncating='post')
        seqEn = to_categorical(seqEn, num_classes=vocab_size_en)
        return np.asarray(seqFr), np.asarray(seqEn)

def seq2seq_model_single(HIDDEN_DIM=128):
    encoder_inputs = tf.keras.layers.Input(shape=(string_max_length))
    encoder_embedding = tf.keras.layers.Embedding(vocab_size_fr, HIDDEN_DIM, input_length=string_max_length)(encoder_inputs)

    encoder_LSTM = tf.keras.layers.LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_LSTM = tf.keras.layers.LSTM(HIDDEN_DIM, return_sequences=True)
    decoder_output = decoder_LSTM(encoder_outputs, initial_state=[state_h, state_c])
    
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size_en, activation='softmax'))(decoder_output)
    model = tf.keras.Model(encoder_inputs, outputs)
    
    return model
    

if os.path.exists(model_name) and reuse_model:
    print("Load: " + model_name)
    model = tf.keras.models.load_model(model_name)
else:
        model = seq2seq_model_single(HIDDEN_DIM)
if train_model:
        tf.keras.utils.plot_model(model, to_file='model_sentence.png', show_shapes=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(MySequence(x_train.to_numpy(), y_train.to_numpy(), batch_size), validation_data=(MySequence(x_test.to_numpy(), y_test.to_numpy(), batch_size)), epochs=epochs, batch_size=batch_size, shuffle=True)
        model.save(model_name)

def decode_sequence(input_seq):
    output_tokens = model.predict(input_seq, verbose=0)
    sampled_token_index = np.argmax(output_tokens, axis=-1)
    return tokenizerEn.sequences_to_texts(sampled_token_index)[0]

print("Test translation")
samples = train_data.sample(5)
for test_fr, test_en in list(zip(samples['fr'].to_list(), samples['en'].to_list())):
    print("fr: ", test_fr)
    test_fr_vec = tokenizerFr.texts_to_sequences([test_fr])
    test_fr_vec = pad_sequences(test_fr_vec, maxlen=string_max_length, truncating='post')
    decoded_sentence = decode_sequence(test_fr_vec)
    print("decoded_sentence: ", decoded_sentence)