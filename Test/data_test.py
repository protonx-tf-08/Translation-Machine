import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data

if __name__ == '__main__':
    train_path = r"D:\Python\Translation_Machine\Data\train.csv"
    valid_path = r"D:\Python\Translation_Machine\Data\valid.csv"
    test_path = r"D:\Python\Translation_Machine\Data\test.csv"
    dataset = data.Data_Preprocessing(train_path, valid_path, test_path)
    max_input_length = 15
    max_target_length = 15
    batch_size = 64

    train_dataset, val_dataset, test_dataset, input_tokenizer, target_tokenizer = dataset.data_process(
        max_input_length, max_target_length, batch_size)
    print('done')
    print(train_dataset)

    print(list(train_dataset.take(1)))
