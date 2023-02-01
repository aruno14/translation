import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

reuse_model=True
train_model=False
model_name="model_seq2seq_save/"
string_max_length=10
batch_size = 32
epochs = 16
HIDDEN_DIM = 1024

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
        batch_x_string = []
        batch_x_sub = []
        batch_y_string = []

        for i in range(0, self.batch_size):
            seqFr = tokenizerFr.texts_to_sequences([self.x_string[i]])
            seqFr = pad_sequences(seqFr, maxlen=string_max_length, truncating='post')[0]
            seqEn = tokenizerEn.texts_to_sequences([self.y_string[i]])[0]
            for y in range(0, len(seqEn)-1):
                decoder_input = np.zeros((string_max_length))
                decoder_input[-y-1:] = seqEn[:min(y+1,string_max_length)]
                decoder_output = np.zeros((vocab_size_en))
                decoder_output[seqEn[y+1]] = 1
                batch_x_string.append(seqFr)
                batch_x_sub.append(decoder_input)
                batch_y_string.append(decoder_output)
        return [np.asarray(batch_x_string), np.asarray(batch_x_sub)], np.asarray(batch_y_string)

def seq2seq_model(HIDDEN_DIM=128):
    encoder_inputs = tf.keras.layers.Input(shape=(string_max_length))
    encoder_embedding = tf.keras.layers.Embedding(vocab_size_fr, HIDDEN_DIM, input_length=string_max_length)(encoder_inputs)

    decoder_inputs = tf.keras.layers.Input(shape=(string_max_length))
    decoder_embedding = tf.keras.layers.Embedding(vocab_size_en, HIDDEN_DIM, input_length=string_max_length)(decoder_inputs)

    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(HIDDEN_DIM, return_state=True)(encoder_embedding)    
    decoder_output = tf.keras.layers.LSTM(HIDDEN_DIM)(decoder_embedding, initial_state=[state_h, state_c])
    
    outputs = tf.keras.layers.Dense(vocab_size_en, activation='softmax')(decoder_output)
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
    
    return model

if os.path.exists(model_name) and reuse_model:
    print("Load: " + model_name)
    model = tf.keras.models.load_model(model_name)
else:
    model = seq2seq_model(HIDDEN_DIM)
if train_model:
    tf.keras.utils.plot_model(model, to_file='model_seq2seq.png', show_shapes=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(MySequence(x_train.to_numpy(), y_train.to_numpy(), batch_size), validation_data=(MySequence(x_test.to_numpy(), y_test.to_numpy(), batch_size)), epochs=epochs, batch_size=batch_size, shuffle=True)
    model.save(model_name)

def decode_sequence(input_seq):
    decoded_sentence = tokenizerEn.texts_to_sequences(["<BOS>"])[0]
    while len(decoded_sentence) < string_max_length:
        sequence = pad_sequences([decoded_sentence], maxlen=string_max_length, truncating='post')
        output_tokens = model.predict([input_seq, sequence], verbose=0)
        sampled_token_index = np.argmax(output_tokens, axis=-1)[0]
        decoded_sentence.append(sampled_token_index)
    return tokenizerEn.sequences_to_texts([decoded_sentence])[0]

print("Test translation")
samples = train_data.sample(15)
for test_fr, test_en in list(zip(samples['fr'].to_list(), samples['en'].to_list())):
    print("fr: ", test_fr)
    test_fr_vec = tokenizerFr.texts_to_sequences([test_fr])
    test_fr_vec = pad_sequences(test_fr_vec, maxlen=string_max_length, truncating='post')
    decoded_sentence = decode_sequence(test_fr_vec)
    print("decoded_sentence: ", decoded_sentence)