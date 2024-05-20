import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model
import tensorflow as tf

def test_model():
    d_model = 512
    num_heads = 8
    num_layers = 6
    dff = 2048
    input_vocab_size = 8500
    target_vocab_size = 8000
    pe_input = 10000
    pe_target = 6000
    transformer = model.Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size, pe_input, pe_target)
    result = transformer(inputs=(tf.random.uniform(
        (64, 38)), tf.random.uniform((64, 37))))

    print(result.shape)


test_model()
