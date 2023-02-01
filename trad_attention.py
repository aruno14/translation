import os
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Code based on https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
tf.config.set_visible_devices([], 'GPU')

reuse_model=False
train_model=True
encoder_name="necoder_save/"
decoder_name="decoder_save/"
string_max_length=8
batch_size = 32
epochs = 2
HIDDEN_DIM = 512
teacher_forcing_ratio = 0.6

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
        return seqFr, seqEn

def seq2seq_model_attention(HIDDEN_DIM=128):
    encoder_input = tf.keras.layers.Input(shape=(string_max_length))
    encoder_embedding = tf.keras.layers.Embedding(vocab_size_fr, HIDDEN_DIM, input_length=string_max_length)(encoder_input)
    encoder_outputs = tf.keras.layers.LSTM(HIDDEN_DIM, return_sequences=True)(encoder_embedding)

    #encoder_prev_state = tf.keras.layers.Input(shape=(HIDDEN_DIM*2))
    #state_h, state_c = tf.split(encoder_prev_state, num_or_size_splits=2, axis=1)
    #encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding, initial_state=[state_h, state_c])
    #encoder_LSTM = tf.keras.layers.LSTM(HIDDEN_DIM, return_state=True, return_sequences=True, name="encoder_lstm")
    #state_h, state_c = tf.split(encoder_prev_state, num_or_size_splits=2, axis=1)
    #encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding, initial_state=[state_h, state_c])
    #encoder_state = tf.keras.layers.Concatenate(axis=-1)([state_h, state_c])
    
    #encoder_model = tf.keras.Model([encoder_input, encoder_prev_state], [encoder_outputs, encoder_state])
    encoder_model = tf.keras.Model(encoder_input, encoder_outputs)
    tf.keras.utils.plot_model(encoder_model, to_file='model_encoder.png', show_shapes=True)

    ###################################################################################

    decoder_input = tf.keras.layers.Input(shape=(string_max_length), name="input_decoder")
    decoder_embedding = tf.keras.layers.Embedding(vocab_size_fr, HIDDEN_DIM, input_length=string_max_length, name="decoder_embedding")(decoder_input)
    embedded = tf.keras.layers.Dropout(0.2)(decoder_embedding)

    decoder_state_input = tf.keras.layers.Input(shape=(HIDDEN_DIM*2), name="input_decoder_previous_state")
    decoder_encoder_state = tf.keras.layers.Input(shape=(string_max_length, HIDDEN_DIM), name="input_decoder_encoder_state")
    
    concat1 = tf.keras.layers.Concatenate(axis=-1)([embedded[:, -1], decoder_state_input])
    attn_weights = tf.keras.layers.Dense(string_max_length, activation='softmax')(concat1)

    attn_applied = tf.keras.layers.Attention()([tf.expand_dims(decoder_encoder_state, axis=-1), tf.expand_dims(attn_weights, axis=-1)])
    output = tf.keras.layers.Concatenate(axis=-1)([embedded, tf.squeeze(attn_applied, axis=-1)])
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(HIDDEN_DIM, activation='relu'))(output)

    decoder_LSTM = tf.keras.layers.LSTM(HIDDEN_DIM, return_state=True)
    state_h, state_c = tf.split(decoder_state_input, num_or_size_splits=2, axis=1)
    decoder_output, state_h, state_c = decoder_LSTM(output, initial_state=[state_h, state_c])
    hidden = tf.keras.layers.Concatenate(axis=-1)([state_h, state_c])
    output = tf.keras.layers.Dense(vocab_size_en, activation='softmax')(decoder_output)

    decoder_model = tf.keras.Model([decoder_input, decoder_encoder_state, decoder_state_input], [output, hidden])
    tf.keras.utils.plot_model(decoder_model, to_file='model_decoder.png', show_shapes=True)
    
    return encoder_model, decoder_model

if reuse_model and os.path.exists(encoder_name) and os.path.exists(decoder_name):
    encoder_model = tf.keras.models.load_model(encoder_name)
    decoder_model = tf.keras.models.load_model(decoder_name)
else:
    encoder_model, decoder_model = seq2seq_model_attention(HIDDEN_DIM)
if train_model:
    sequence_train = MySequence(x_train.to_numpy(), y_train.to_numpy(), batch_size)
    optimizer_encoder = tf.keras.optimizers.Adam()
    optimizer_decoder = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for idx in range(0, len(sequence_train)):
            batch_x_string, batch_y_string = sequence_train[idx]
            loss_value=0
            force_training = True if random.random() < teacher_forcing_ratio else False
            with tf.GradientTape(persistent=True) as tape:
                encoder_outputs = encoder_model(batch_x_string, training=True)
                for i in range(0, batch_size):
                    with tape.stop_recording():
                        hidden = tf.zeros((1, HIDDEN_DIM*2))
                        decoder_input = np.zeros((1, string_max_length))
                        previous_output = None
                    for y in range(0, len(batch_y_string[i])-1):
                        with tape.stop_recording():
                            correct = np.zeros(vocab_size_en)
                            correct[batch_y_string[i][y+1]]=1
                            if force_training or y==0:
                                decoder_input[0][-y-1:] = batch_y_string[i][:min(y+1, string_max_length)]
                            else:
                                decoder_input[0][:-1] = decoder_input[0][1:]
                                decoder_input[0][-1] = np.argmax(previous_output)
                        
                        decoder_outputs, hidden = decoder_model([decoder_input, tf.slice(encoder_outputs, [i, 0, 0], size=[1, encoder_outputs.shape[1], encoder_outputs.shape[2]]), hidden], training=True)
                        previous_output = decoder_outputs[0]
                        
                        loss_value = loss_value + loss_fn(correct, previous_output)
                        #if loss_value < 0.001:
                        #    print(loss_value)
                        #    print("decoder_input", tokenizerEn.sequences_to_texts(decoder_input))
                        #    print("next", tokenizerEn.sequences_to_texts([[batch_y_string[i][y+1]]]))
                        #    print(decoder_outputs[0], np.argmax(decoder_outputs[0]), tokenizerEn.sequences_to_texts([[np.argmax(decoder_outputs[0])]]))

            grads_encoder = tape.gradient(loss_value, encoder_model.trainable_weights)
            grads_decoder = tape.gradient(loss_value, decoder_model.trainable_weights)

            optimizer_encoder.apply_gradients(zip(grads_encoder, encoder_model.trainable_weights))
            optimizer_decoder.apply_gradients(zip(grads_decoder, decoder_model.trainable_weights))
            del tape

            # Log every 5 batches.
            if idx % 5 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (idx, float(loss_value)))
                print("Seen so far: %s samples" % ((idx + 1) * batch_size))
    encoder_model.save(encoder_name)
    decoder_model.save(decoder_name)

def decode_sequence(input_seq):
    encoder_output = encoder_model([input_seq])
    decoded_sentence = tokenizerEn.texts_to_sequences(["<BOS>"])[0]
    hidden = tf.zeros((1, HIDDEN_DIM*2))
    while len(decoded_sentence) < string_max_length:
        sequence = pad_sequences([decoded_sentence], maxlen=string_max_length, truncating='post')
        decoder_outputs, hidden = decoder_model([sequence, encoder_output, hidden])
        sampled_token_index = np.argmax(decoder_outputs, axis=-1)[0]
        decoded_sentence.append(sampled_token_index)
    return tokenizerEn.sequences_to_texts([decoded_sentence])[0]

print("Test translation")
samples = train_data.sample(5)
for test_fr, test_en in list(zip(samples['fr'].to_list(), samples['en'].to_list())):
    print("fr: ", test_fr)
    test_fr_vec = tokenizerFr.texts_to_sequences([test_fr])
    test_fr_vec = pad_sequences(test_fr_vec, maxlen=string_max_length, truncating='post')
    decoded_sentence = decode_sequence(test_fr_vec)
    print("decoded_sentence: ", decoded_sentence)