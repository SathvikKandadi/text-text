import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Dot, Activation
from tensorflow.keras.models import Model

class Translator:
    def __init__(self, eng_tokenizer, deu_tokenizer, max_len=20, 
                 embedding_dim=512, units=512):  # Increased dimensions
        self.eng_tokenizer = eng_tokenizer
        self.deu_tokenizer = deu_tokenizer
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.units = units
        
        # Build models
        self.full_model, self.encoder_model, self.decoder_model = self._build_models()

    def _build_models(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=len(self.eng_tokenizer.word_index) + 1,
                                   output_dim=self.embedding_dim,
                                   mask_zero=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        
        # Bidirectional LSTM with half the units (will double after concatenation)
        encoder_lstm = tf.keras.layers.Bidirectional(
            LSTM(self.units // 2, return_sequences=True, return_state=True,
                 dropout=0.2, recurrent_dropout=0.2)
        )
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedded)
        
        # Combine states
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder with matching dimensions
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=len(self.deu_tokenizer.word_index) + 1,
                                   output_dim=self.embedding_dim,
                                   mask_zero=True)
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(self.units, return_sequences=True, return_state=True,
                          dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

        # Attention mechanism
        attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
        attention = Activation('softmax')(attention)
        context = Dot(axes=[2, 1])([attention, encoder_outputs])
        decoder_concat = Concatenate(axis=-1)([decoder_outputs, context])

        # Dense layer
        decoder_dense = Dense(len(self.deu_tokenizer.word_index) + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_concat)

        # Full training model
        full_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        full_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Inference models
        encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # Decoder inference model
        decoder_inputs_inf = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.units,))
        decoder_state_input_c = Input(shape=(self.units,))
        encoder_outputs_inf = Input(shape=(None, self.units))

        # Reuse layers properly
        decoder_embedded_inf = decoder_embedding(decoder_inputs_inf)
        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
            decoder_embedded_inf, 
            initial_state=[decoder_state_input_h, decoder_state_input_c]
        )

        # Custom attention for inference
        attention_inf = Dot(axes=[2, 2])([decoder_outputs_inf, encoder_outputs_inf])
        attention_inf = Activation('softmax')(attention_inf)
        context_inf = Dot(axes=[2, 1])([attention_inf, encoder_outputs_inf])
        decoder_concat_inf = Concatenate(axis=-1)([decoder_outputs_inf, context_inf])
        
        decoder_outputs_inf = decoder_dense(decoder_concat_inf)

        decoder_model = Model(
            inputs=[decoder_inputs_inf, encoder_outputs_inf, decoder_state_input_h, decoder_state_input_c],
            outputs=[decoder_outputs_inf, state_h_inf, state_c_inf]
        )

        return full_model, encoder_model, decoder_model