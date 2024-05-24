import tensorflow as tf
import pandas as pd
from datasets import Dataset, load_from_disk
import underthesea
import pickle
import re
import os
import sys


class Data_Preprocessing:
    def __init__(self, train_path, val_path, test_path, type_data='csv'):
        if (type_data == 'csv'):
            self.train_dataset = pd.read_csv(train_path)
            self.val_dataset = pd.read_csv(val_path)
            self.test_dataset = pd.read_csv(test_path)
        if (type_data == 'arrow'):
            self.train_dataset = self.load_dataset(train_path)
            self.val_dataset = self.load_dataset(val_path)
            self.test_dataset = self.load_dataset(test_path)

    def tokenizer(self, dataset, language='en', tokenizer_en=None, tokenizer_vi=None):
        if language == "vi":
            dataset = [underthesea.word_tokenize(
                text, format='text') for text in dataset]
            tokenizer = tokenizer_vi
        else:
            tokenizer = tokenizer_en
        if tokenizer is None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(
                filters='', oov_token='<unk>')
            tokenizer.fit_on_texts(dataset)
        tensor = tokenizer.texts_to_sequences(dataset)
        return tensor, tokenizer

    def preprocess_sentence(self, sentence):
        sentence = str(sentence).replace("_", " ")
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<start>', sentence, '<end>')
        return sentence

    def split_envi(self, examples):
        examples['en'] = examples['en'].map(self.preprocess_sentence)
        examples['vi'] = examples['vi'].map(self.preprocess_sentence)
        inputs = [ex for ex in examples['en']]
        targets = [ex for ex in examples['vi']]
        return inputs, targets

    def padding(self, tensor, max_length):
        tensor = tf.keras.utils.pad_sequences(
            tensor, padding='post', maxlen=max_length)
        return tensor

    def preprocess(self, examples, max_input_length, max_target_length, tokenizer_en=None, tokenizer_vi=None):
        inputs, targets = self.split_envi(examples)
        input_tensor, input_tokenizer = self.tokenizer(
            dataset=inputs, language="en", tokenizer_en=tokenizer_en)
        target_tensor, target_tokenizer = self.tokenizer(
            dataset=targets, language='vi', tokenizer_vi=tokenizer_vi)
        input_tensor = self.padding(
            input_tensor, max_length=max_input_length)
        target_tensor = self.padding(
            target_tensor, max_length=max_target_length)
        return input_tensor, target_tensor, input_tokenizer, target_tokenizer

    # def convert_tfdataset(self, input_tensor, target_tensor, batch_size):
    #     dataset = tf.data.Dataset.from_tensor_slices(
    #         (input_tensor, target_tensor))
    #     dataset = dataset.shuffle(10000).batch(
    #         batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    #     return dataset

    def split_input_target(self, en, vi, path_save):
        input_en = []
        input_vi = []
        target_vi = []

        # Tạo các danh sách riêng biệt cho từng cột
        for f1, f2, label in zip(en, vi[:, :-1], vi[:, 1:]):
            input_en.append(f1)
            input_vi.append(f2)
            target_vi.append(label)

        # Tạo từ điển dữ liệu trực tiếp từ các danh sách đã tạo
        reformatted_data_dict = {
            'input_en': input_en,
            'input_vi': input_vi,
            'target_vi': target_vi
        }

        # Tạo Dataset từ từ điển
        hf_dataset = Dataset.from_dict(reformatted_data_dict)
        hf_dataset.save_to_disk(path_save)
        return hf_dataset

    def data_process(self, max_input_length, max_target_length, path_train, path_test, path_valid):
        train_dataset_inptensor, train_datasetout_tensor, input_tokenizer, target_tokenizer = self.preprocess(
            self.train_dataset, max_input_length, max_target_length)
        val_dataset_inptensor, val_datasetout_tensor, _, _ = self.preprocess(self.val_dataset,
                                                                             max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)
        test_dataset_inptensor, test_datasetout_tensor, _, _ = self.preprocess(self.test_dataset,
                                                                               max_input_length, max_target_length, tokenizer_en=input_tokenizer, tokenizer_vi=target_tokenizer)

        train_dataset = self.split_input_target(
            train_dataset_inptensor, train_datasetout_tensor, path_train)
        val_dataset = self.split_input_target(
            val_dataset_inptensor, val_datasetout_tensor, path_valid)
        test_dataset = self.split_input_target(
            test_dataset_inptensor, test_datasetout_tensor, path_test)

        with open(r'Tokenizer\en_tokenizer.json', 'w') as f:
            f.write(input_tokenizer.to_json())

        with open(r'Tokenizer\vi_tokenizer.json', 'w') as f:
            f.write(target_tokenizer.to_json())

        return train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer

    def load_dataset(self, path_load):
        if not os.path.exists(path_load):
            print(f"File {path_load} does not exist.")
            sys.exit(1)  # Kết thúc chương trình với mã lỗi 1
        else:
            print(f"File {path_load} exists.")
        return load_from_disk(path_load)

    def load_tokenizer(self, tokenizer_en_path, tokenizer_vi_path):
        if not os.path.exists(tokenizer_en_path):
            print(f"File {tokenizer_en_path} does not exist.")
            sys.exit(1)

        if not os.path.exists(tokenizer_vi_path):
            print(f"File {tokenizer_vi_path} does not exist.")
            sys.exit(1)

        with open(tokenizer_en_path) as f:
            tokenizer_en = tf.keras.preprocessing.text.tokenizer_from_json(
                f.read())

        with open(tokenizer_vi_path) as f:
            tokenizer_vi = tf.keras.preprocessing.text.tokenizer_from_json(
                f.read())

        return tokenizer_en, tokenizer_vi

    def load_data_tokenizer(self, tokenizer_en_path, tokenizer_vi_path, batch_size=32, shuffle=True):
        tokenizer_en, tokenizer_vi = self.load_tokenizer(
            tokenizer_en_path, tokenizer_vi_path)
        return self.convert_to_tf_dataset(self.train_dataset, batch_size, shuffle), self.convert_to_tf_dataset(self.val_dataset, batch_size, shuffle), self.convert_to_tf_dataset(self.test_dataset, batch_size, shuffle), tokenizer_en, tokenizer_vi

    def convert_to_tf_dataset(self, hf_dataset, batch_size=32, shuffle=True):
        def encode(examples):
            return {
                'input_en': tf.constant(examples['input_en']),
                'input_vi': tf.constant(examples['input_vi']),
                'target_vi': tf.constant(examples['target_vi'])
            }

        # Map the dataset to encode each example
        tf_dataset = hf_dataset.map(encode)

        # Convert to a tf.data.Dataset
        tf_dataset = tf_dataset.to_tf_dataset(
            columns=['input_en', 'input_vi'],
            label_cols='target_vi',
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=None  # Use the default collate function
        )

        return tf_dataset


class Data_Predict:
    def __init__(self, data_path, tokenizer_en_path, tokenizer_vi_path):
        self.data = self.load_data(data_path)
        self.tokenizer_en, self.tokenizer_vi = self.load_tokenizer(
            tokenizer_en_path, tokenizer_vi_path)

    def load_tokenizer(self, tokenizer_en_path, tokenizer_vi_path):
        with open(tokenizer_en_path, 'rb') as handle:
            tokenizer_en = pickle.load(handle)

        with open(tokenizer_vi_path, 'rb') as handle:
            tokenizer_vi = pickle.load(handle)
        return tokenizer_en, tokenizer_vi

    def load_data(self, data_path):
        lines_predict = []
        with open(data_path, 'r') as file:
            # Đọc từng dòng của file và lưu vào mảng
            for line in file:
                # Thêm dòng vào mảng sau khi loại bỏ các ký tự trắng thừa
                lines_predict.append(line.strip())
        return lines_predict

    def preprocess_sentence(self, sentence):
        sentence = str(sentence)
        sentence = sentence.lower().strip()
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = sentence.strip()

        # Add start and end token
        sentence = '{} {} {}'.format('<start>', sentence, '<end>')
        return sentence

    def predict_data_preprocessing(self, max_length):
        lines_predict_preprocessed = [self.preprocess_sentence(
            sentence) for sentence in self.data]
        input_en = self.tokenizer_en.texts_to_sequences(
            lines_predict_preprocessed)
        input_en = tf.keras.utils.pad_sequences(
            input_en, padding='post', maxlen=max_length)
        input_tensor = tf.convert_to_tensor(input_en, dtype=tf.int64)
        return input_tensor, self.tokenizer_en

    def detokenizer(self, tensor):
        detokenized_texts = self.tokenizer_vi.sequences_to_texts(tensor)
        return [sentence.replace('<start>', '').replace('<end>', '').replace('<unk>', '').replace('_', ' ') for sentence in detokenized_texts], self.tokenizer_vi
