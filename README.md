# LSTM-Based Generative Text Model — Next Word Prediction with Early Stopping

Trains an LSTM network on Shakespeare's *Hamlet* to predict the next word in a given sequence. The trained model is served through a Streamlit web app where you type a phrase and get the predicted next word back instantly.

---

## How it works

**Dataset:** The full text of *Hamlet* pulled from NLTK's Gutenberg corpus and saved as `hamlet.txt`. The entire text is lowercased and tokenized before training.

**Preprocessing:**
- Text tokenized using Keras `Tokenizer`, building a word index across the full vocabulary
- N-gram sequences generated line by line — for each line, every prefix of length 2 up to the full line is added as a training sequence
- All sequences padded to `max_sequence_len` with pre-padding
- Last word of each sequence is the label, one-hot encoded across the full vocabulary
- 80/20 train/test split

**Model Architecture:**

```
Embedding(vocab_size, 100, input_length=max_sequence_len-1)
    ↓
LSTM(150, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(100)
    ↓
Dense(vocab_size, activation='softmax')
```

- Loss: Categorical Crossentropy
- Optimizer: Adam
- Early Stopping: monitors `val_loss`, patience=3, restores best weights
- Max epochs: 50

> A GRU variant is also included in the notebook (commented out) as an alternative architecture to experiment with.

**Inference:** Input text is tokenized and padded to match the training sequence length. The model outputs a probability distribution over the full vocabulary and `argmax` picks the predicted word.

---

## Streamlit App

```bash
streamlit run app.py
```

Type any sequence of words into the input box, hit **Predict Next Word**, and the app returns the predicted next word using the saved model and tokenizer.

Example input: `To be or not to` → predicted next word based on Hamlet's vocabulary

---

## Stack

Python · TensorFlow 2.20 · Keras · NLTK · scikit-learn · NumPy · Streamlit · pickle

---

## Project Layout

```
├── app.py                   # Streamlit web app
├── experiments.ipynb        # Full pipeline: preprocessing, training, inference
├── hamlet.txt               # Shakespeare's Hamlet (training corpus)
├── next_word_lstm.keras     # Saved LSTM model
├── next_word_lstm.h5        # Saved model (alternative format)
├── tokenizer.pickle         # Saved Keras tokenizer
├── requirements.txt
└── .devcontainer/
    └── devcontainer.json
```

---

## Run it locally

```bash
git clone https://github.com/JainulTrivedi55555/LSTM-Based-Generative-Text-Model-Next-Word-Prediction-with-Early-Stopping.git
cd LSTM-Based-Generative-Text-Model-Next-Word-Prediction-with-Early-Stopping
pip install -r requirements.txt
streamlit run app.py
```
