import csv, os, string, json
import tensorflow as tf
from tensorflow import keras
import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_TEXT_SIZE = 255

def punctuation_pairs():
    return [ 
        ("!", " _explanation_mark_ "), ("?", " _question_mark_ "), ("“", "\""), ("’", " ’ "),\
        ("\n", ""), ("\t", ""), ("$", " _dollar_ "), ("_id", "_iid"), ("'", " ' "), (".", " _period_ ")\
    ]

def listify_txt(txt: str) -> list:
    """
    Returns a list of words from string txt and sets all letters to be lowercase
    """
    for replacement in punctuation_pairs():
        txt = txt.replace(replacement[0], replacement[1])
    for punctuation in string.punctuation:
        txt = txt.replace(punctuation, " " + punctuation + " ")
    text_list = txt.lower().strip().split(" ")
    return list(filter(lambda elem: elem != "", text_list))

def delistify_txt(txt_lst: list) -> str:
    txt = " ".join(txt_lst)
    # print(txt)
    for replacement in filter(lambda x: x[1] != "", punctuation_pairs()):
        txt = txt.replace(replacement[1], replacement[0])
    for punctuation in string.punctuation:
        txt = txt.replace( " " + punctuation, punctuation) 
    return txt

def fill_word_index(word_index: dict, word_list: list):
    for word in word_list:
        if word not in word_index:
            word_index[word] = len(word_index)

def wordlist_to_array(word_index: dict, word_list: list, max_txt_size=255) -> np.ndarray:
    encoded = np.array([word_index['<START>']] + [word_index.get(w, word_index['<UNK>']) for w in word_list], dtype=np.int64)
    return keras.preprocessing.sequence.pad_sequences([encoded], value=word_index["<PAD>"], padding="post", maxlen=max_txt_size)[0]

def load_processed_data(max_txt_size=255):
    word_index = {'<PAD>': 0, '<START>': 1, '<UNK>': 2, '<UNUSED>': 3}
    x_array = []
    y_array = []

    with open(os.path.join(data_dir, "conv_data.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            saying_lst = listify_txt(row[0])
            fill_word_index(word_index, saying_lst)
            x_array.append(wordlist_to_array(word_index, saying_lst, max_txt_size=max_txt_size))
            response_lst = listify_txt(row[0])
            fill_word_index(word_index, response_lst)
            y_array.append(wordlist_to_array(word_index, response_lst, max_txt_size=max_txt_size))
    with open(os.path.join(data_dir, "word_list.json"), "w") as f:
        json.dump(word_index, f)
    return tf.convert_to_tensor(x_array, dtype=tf.int64), tf.convert_to_tensor(y_array, dtype=tf.int64), word_index

def load_word_index():
    with open(os.path.join(data_dir, "word_list.json"), "r") as f:
        return json.load(f)

def reverse_word_index(word_index: dict):
    return {item[1]:item[0] for item in word_index.items()}

def deprocess_txt(nums: list, word_index: dict) -> str:
    rev_indx = reverse_word_index(word_index)
    s = delistify_txt([rev_indx.get(round(x), "<UNK>") for x in nums])
    return s.replace(rev_indx[0], "").replace(rev_indx[1], "").replace(rev_indx[2], "").replace(rev_indx[3], "")

def preprocess_text(text: str, word_index: dict, max_txt_size=255) -> np.ndarray:
    return wordlist_to_array(word_index, listify_txt(text), max_txt_size=max_txt_size)

def create_and_fit_model(x_train, y_train, embeding_dim: tuple, epochs=7, batch_size=None, validation_data=None) -> keras.Model:
    model: keras.Model = keras.Sequential()
    model.add(keras.layers.Embedding(embeding_dim[0],embeding_dim[1]))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(10000,activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(5000,activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(255,activation="relu"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=1)

    if validation_data == None:
        print('no validation data added')
    else:
        accuracy = model.evaluate(validation_data[0], validation_data[1])[1]
        print(f'Finished training with model with validation accuracy {accuracy}')
        print(f'Will now save model')

    return model

def train_model():
    x_train, y_train, word_index = load_processed_data()
    train_size = len(x_train) - len(x_train)//5
    x_train, y_train, X_val, y_val = x_train[:train_size], y_train[:train_size], x_train[train_size:], y_train[train_size:]
    model = create_and_fit_model(x_train, y_train, embeding_dim=(len(word_index), 32), \
        epochs=400, batch_size=10, validation_data=(X_val, y_val))
    model.save(os.path.join(data_dir, "chatbot_neural_ntwk.h5"), include_optimizer=False)
    del model
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    import sys
    if "train" in sys.argv:
        train_model()
        exit(0)
    word_index = load_word_index()
    model = keras.models.load_model(os.path.join(data_dir, "chatbot_neural_ntwk.h5"), compile=False)
    prediction_nums_arrays: np.ndarray = model.predict(preprocess_text(input("say something:"), word_index))
    for n in prediction_nums_arrays[:6]:
        print("---------------------------------------------------------")
        print(deprocess_txt(n, word_index))