import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# === 0. GPU KONTROLÜ ===
if tf.config.list_physical_devices('GPU'):
    print(" GPU kullanılabilir.")
else:
    print("GPU bulunamadı, CPU ile devam edilecek.")

# === 1. HYPERPARAMETERS ===
BATCH_SIZE = 128
EPOCHS = 10
LATENT_DIM = 256
EMBEDDING_DIM = 64
MAX_TEXT_LEN = 300
MAX_SUMMARY_LEN = 50
VOCAB_SIZE = 30000

# === 2. VERİ YÜKLEME ===
train_path = "./dataset/train.csv"
val_path = "./dataset/validation.csv"

train_df = pd.read_csv(train_path).dropna()
val_df = pd.read_csv(val_path).dropna()

def preprocess(text):
    return str(text).lower().strip()

train_texts = train_df['article'].apply(preprocess)
train_summaries = train_df['highlights'].apply(lambda x: '_START_ ' + preprocess(x) + ' _END_')
val_texts = val_df['article'].apply(preprocess)
val_summaries = val_df['highlights'].apply(lambda x: '_START_ ' + preprocess(x) + ' _END_')

# === 3. TOKENIZER ve PAD ===
text_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
text_tokenizer.fit_on_texts(train_texts)
x_train_seq = text_tokenizer.texts_to_sequences(train_texts)
x_val_seq = text_tokenizer.texts_to_sequences(val_texts)

x_train_pad = pad_sequences(x_train_seq, maxlen=MAX_TEXT_LEN, padding='post')
x_val_pad = pad_sequences(x_val_seq, maxlen=MAX_TEXT_LEN, padding='post')

summary_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
summary_tokenizer.fit_on_texts(train_summaries)
y_train_seq = summary_tokenizer.texts_to_sequences(train_summaries)
y_val_seq = summary_tokenizer.texts_to_sequences(val_summaries)

y_train_pad = pad_sequences(y_train_seq, maxlen=MAX_SUMMARY_LEN, padding='post')
y_val_pad = pad_sequences(y_val_seq, maxlen=MAX_SUMMARY_LEN, padding='post')

y_train_targets = y_train_pad.reshape(y_train_pad.shape[0], y_train_pad.shape[1], 1)
y_val_targets = y_val_pad.reshape(y_val_pad.shape[0], y_val_pad.shape[1], 1)

# === 4. tf.data.Dataset ===
train_ds = tf.data.Dataset.from_tensor_slices(((x_train_pad, y_train_pad), y_train_targets))
train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((x_val_pad, y_val_pad), y_val_targets))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === 5. MODEL MİMARİSİ ===
encoder_inputs = Input(shape=(MAX_TEXT_LEN,))
enc_emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=True)(encoder_inputs)
_, encoder_state = GRU(LATENT_DIM, return_state=True)(enc_emb)

decoder_inputs = Input(shape=(MAX_SUMMARY_LEN,))
dec_emb_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_gru = GRU(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(dec_emb, initial_state=encoder_state)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.summary()

# === 6. EĞİTİM ===
checkpoint_path = "model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

model.fit(train_ds,
          epochs=EPOCHS,
          validation_data=val_ds,
          callbacks=[checkpoint])

# === 7. TOKENIZER KAYIT ===
with open('text_tokenizer.pkl', 'wb') as f:
    pickle.dump(text_tokenizer, f)
with open('summary_tokenizer.pkl', 'wb') as f:
    pickle.dump(summary_tokenizer, f)

print(" Eğitim tamamlandı. Model ve tokenizer'lar kaydedildi.")
