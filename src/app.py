import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Layer
from model import Translator
import os
import pickle

# Load the dataset

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().split('\n')
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    pairs = []
    for line in lines:
        if line and '\t' in line:
            # Split by tab and take only the translation part
            parts = line.split('\t')[0]
            # Split English and German parts
            if '\t' not in parts:
                eng_deu = parts.split('.')
                if len(eng_deu) >= 2:
                    eng = ' '.join(eng_deu[:-1]).strip()  # English
                    deu = eng_deu[-1].strip()  # German
                    # Only keep simple translations
                    if eng and deu and len(eng.split()) < 5 and len(deu.split()) < 5:
                        pairs.append([eng, deu])
    
    # Add manual basic translations to ensure coverage
    basic_translations = [
        ["hello", "hallo"],
        ["hi", "hallo"],
        ["how are you", "wie geht es dir"],
        ["good morning", "guten morgen"],
        ["goodbye", "auf wiedersehen"],
        ["thank you", "danke"],
        ["yes", "ja"],
        ["no", "nein"],
        ["please", "bitte"],
    ]
    pairs.extend(basic_translations * 10)  # Add these multiple times
    
    print("Sample basic translations:")
    for pair in basic_translations:
        print(f"English: {pair[0]}, German: {pair[1]}")
    return pairs

# Clean the data
def clean_data(pairs, max_pairs=2000):  # Reduced max_pairs
    cleaned_pairs = []
    priority_pairs = []
    
    # Define basic words/phrases to prioritize
    basic_words = {'hello', 'hi', 'how', 'good', 'morning', 'thank', 'please', 'yes', 'no'}
    
    for pair in pairs:
        if len(pair) == 2:
            eng, deu = pair[0].lower(), pair[1].lower()
            # Only keep short, simple sentences
            if len(eng.split()) <= 4 and len(deu.split()) <= 4:
                # Prioritize basic phrases
                if any(word in eng.split() for word in basic_words):
                    priority_pairs.append([eng, deu])
                elif len(cleaned_pairs) < max_pairs:
                    cleaned_pairs.append([eng, deu])
    
    # Ensure priority pairs appear multiple times in training
    final_pairs = priority_pairs * 10 + cleaned_pairs
    print(f"Cleaned data contains {len(final_pairs)} pairs (including {len(priority_pairs)} priority pairs)")
    return final_pairs

# Tokenize and pad sequences
def tokenize_and_pad(cleaned_pairs, max_len):
    # Separate English and German sentences
    eng_sentences = [pair[0] for pair in cleaned_pairs]  # English is now input
    deu_sentences = [pair[1] for pair in cleaned_pairs]  # German is now output

    # Tokenize English sentences (input)
    eng_tokenizer = Tokenizer(filters='')
    eng_tokenizer.fit_on_texts(['<start>', '<end>'] + eng_sentences)
    eng_sequences = eng_tokenizer.texts_to_sequences(['<start> ' + sent + ' <end>' for sent in eng_sentences])
    eng_padded = pad_sequences(eng_sequences, maxlen=max_len + 1, padding='post')

    # Tokenize German sentences (output)
    deu_tokenizer = Tokenizer(filters='')
    deu_tokenizer.fit_on_texts(['<start>', '<end>'] + deu_sentences)
    deu_sequences = deu_tokenizer.texts_to_sequences(['<start> ' + sent + ' <end>' for sent in deu_sentences])
    deu_padded = pad_sequences(deu_sequences, maxlen=max_len + 1, padding='post')

    return eng_padded, deu_padded, eng_tokenizer, deu_tokenizer

def main():
    # Load and preprocess data
    file_path = 'deu-eng/deu.txt'
    model_path = 'saved_model'
    
    pairs = load_data(file_path)
    cleaned_pairs = clean_data(pairs, max_pairs=2000)
    max_len = 20
    
    # Prepare the data
    eng_padded, deu_padded, eng_tokenizer, deu_tokenizer = tokenize_and_pad(cleaned_pairs, max_len)
    
    # Create decoder input/output data
    decoder_input_data = deu_padded[:, :-1]
    decoder_output_data = deu_padded[:, 1:]
    
    # Initialize the translator
    translator = Translator(eng_tokenizer, deu_tokenizer, max_len=max_len)
    
    # Check if saved model exists
    if not os.path.exists(f"{model_path}/full_model.keras"):
        print("Training new model...")
        # Train the model
        translator.full_model.fit(
            [eng_padded, decoder_input_data],
            decoder_output_data,
            batch_size=16,
            epochs=100,
            validation_split=0.1,
            shuffle=True
        )
        
        # Save the models
        os.makedirs(model_path, exist_ok=True)
        translator.full_model.save(f"{model_path}/full_model.keras")
        translator.encoder_model.save(f"{model_path}/encoder_model.keras")
        translator.decoder_model.save(f"{model_path}/decoder_model.keras")
        
        # Save tokenizers
        with open(f"{model_path}/eng_tokenizer.pkl", 'wb') as f:
            pickle.dump(eng_tokenizer, f)
        with open(f"{model_path}/deu_tokenizer.pkl", 'wb') as f:
            pickle.dump(deu_tokenizer, f)
    else:
        print("Loading saved model...")
        # Load the models
        translator.full_model = tf.keras.models.load_model(f"{model_path}/full_model.keras")
        translator.encoder_model = tf.keras.models.load_model(f"{model_path}/encoder_model.keras")
        translator.decoder_model = tf.keras.models.load_model(f"{model_path}/decoder_model.keras")
        
        # Load tokenizers
        with open(f"{model_path}/eng_tokenizer.pkl", 'rb') as f:
            eng_tokenizer = pickle.load(f)
        with open(f"{model_path}/deu_tokenizer.pkl", 'rb') as f:
            deu_tokenizer = pickle.load(f)

    # Test cases
    test_phrases = [
        "hello",
        "hi",
        "how are you",
        "good morning",
        "thank you",
        "please",
        "yes",
        "no",
        "what is your name",
        "goodbye",
        "good night",
        "see you later",
        "nice to meet you",
        "have a good day"
    ]
    
    print("\nTesting basic translations:")
    for phrase in test_phrases:
        translated = translate(phrase, eng_tokenizer, deu_tokenizer, translator, max_len)
        print(f"English: {phrase}")
        print(f"German: {translated}\n")

    # Interactive translation
    print("\nEnter 'quit' to exit")
    while True:
        user_input = input("\nEnter English text to translate: ").strip()
        if user_input.lower() == 'quit':
            break
        
        translated = translate(user_input, eng_tokenizer, deu_tokenizer, translator, max_len)
        print(f"German translation: {translated}")

# Move translate function outside of main
def translate(sentence, eng_tokenizer, deu_tokenizer, translator, max_len, max_length=50):
    sentence = sentence.lower()
    sequence = eng_tokenizer.texts_to_sequences(['<start> ' + sentence + ' <end>'])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    encoder_outputs, state_h, state_c = translator.encoder_model.predict(padded_sequence)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = deu_tokenizer.word_index['<start>']
    states_value = [state_h, state_c]

    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition and len(decoded_sentence.split()) < max_length:
        output_tokens, state_h, state_c = translator.decoder_model.predict(
            [target_seq, encoder_outputs] + states_value
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            continue
            
        sampled_word = deu_tokenizer.index_word.get(sampled_token_index, '')
        
        if sampled_word == '<end>':
            stop_condition = True
        else:
            decoded_sentence += sampled_word + ' '

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [state_h, state_c]

    return decoded_sentence.strip()

if __name__ == "__main__":
    main()