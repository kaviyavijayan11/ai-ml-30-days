import pandas as pd
import numpy as np
import re
import string
print("NLP - TEXT PREPROCESSING")
reviews = [
    "This product is AMAZING!!! I love it so much!",
    "Terrible quality. Waste of money :(",
    "It's okay, nothing special but works fine.",
    "Best purchase ever!!! Highly recommend to everyone",
    "Very disappointed. Product broke after 2 days."
]

labels = [1, 0, 1, 1, 0]
print("\nOriginal Reviews:")
for i, review in enumerate(reviews):
    print(f"{i+1}. {review}")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaned Reviews:")
cleaned_reviews = [clean_text(r) for r in reviews]
for i, review in enumerate(cleaned_reviews):
    print(f"{i+1}. {review}")
print("\nWord Count Analysis:")
for i, review in enumerate(cleaned_reviews):
    words = review.split()
    print(f"Review {i+1}: {len(words)} words")
all_words = " ".join(cleaned_reviews).split()
unique_words = set(all_words)
print(f"\nTotal words: {len(all_words)}")
print(f"Unique words: {len(unique_words)}")
print(f"Vocabulary: {sorted(unique_words)}")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("TOKENIZATION")

reviews = [
    "This product is amazing i love it so much",
    "Terrible quality waste of money",
    "Its okay nothing special but works fine",
    "Best purchase ever highly recommend to everyone",
    "Very disappointed product broke after 2 days"
]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
word_index = tokenizer.word_index
print(f"\nVocabulary size: {len(word_index)}")
print(f"\nSample word-to-number mapping:")
for word, num in list(word_index.items())[:15]:
    print(f"  '{word}' -> {num}")
sequences = tokenizer.texts_to_sequences(reviews)
print(f"\nText to Sequences:")
for i, (review, seq) in enumerate(zip(reviews, sequences)):
    print(f"\nReview {i+1}: {review}")
    print(f"Sequence: {seq}")
max_length = 10
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
print(f"\nPadded Sequences (max_length={max_length}):")
for i, p in enumerate(padded):
    print(f"Review {i+1}: {p}")

from tensorflow.keras.layers import Embedding

print("WORD EMBEDDINGS")

print("""
What are Word Embeddings?

Simple approach (one-hot encoding):
"king"  -> [1,0,0,0,0,...]
"queen" -> [0,1,0,0,0,...]
"apple" -> [0,0,1,0,0,...]
Problem: No relationship captured!

Embeddings approach:
"king"  -> [0.2, 0.8, 0.1, 0.5, ...]
"queen" -> [0.3, 0.7, 0.2, 0.4, ...]  (similar to king!)
"apple" -> [0.9, 0.1, 0.8, 0.1, ...]  (different!)

Embeddings capture MEANING:
king - man + woman = queen (famous example!)
""")

vocab_size = 100
embedding_dim = 16
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=10
)

print(f"Embedding Layer:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Embedding dimensions: {embedding_dim}")
print(f"  Each word becomes: {embedding_dim}-dimensional vector")
sample_input = padded[:1]
print(f"\nInput (word numbers): {sample_input[0]}")
embedding_output = embedding_layer(sample_input)
print(f"Output shape: {embedding_output.shape}")
print(f"(1 review, 10 words, {embedding_dim} dimensions each)")
print(f"\nFirst word embedding (first 5 values):")
print(embedding_output[0][0][:5].numpy())

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D

print("SENTIMENT ANALYSIS MODEL")
reviews = [
    "amazing product love it",
    "terrible quality waste money",
    "okay nothing special works fine",
    "best purchase ever highly recommend",
    "disappointed product broke quickly",
    "excellent value great quality",
    "worst experience never buying again",
    "good product satisfied with purchase",
    "poor quality not worth price",
    "fantastic amazing wonderful product",
    "bad experience regret buying",
    "happy with this purchase good",
    "awful product complete waste",
    "perfect exactly what needed",
    "horrible dont waste your money"
]

labels = [1,0,1,1,0,1,0,1,0,1,0,1,0,1,0]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded = pad_sequences(sequences, maxlen=10, padding='post')

X = np.array(padded)
y = np.array(labels)
model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=10),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture:")
model.summary()

print("\nTraining model...")
history = model.fit(X, y, epochs=50, verbose=0)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nTraining Accuracy: {acc*100:.2f}%")
new_reviews = [
    "amazing quality highly recommend",
    "terrible waste of money",
    "okay product nothing special"
]

new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded = pad_sequences(new_sequences, maxlen=10, padding='post')

predictions = model.predict(new_padded, verbose=0)

print("\nNew Review Predictions:")
for review, pred in zip(new_reviews, predictions):
    sentiment = "Positive" if pred[0] > 0.5 else "Negative"
    confidence = pred[0] if pred[0] > 0.5 else 1 - pred[0]
    print(f"'{review}'")
    print(f"  Sentiment: {sentiment}, Confidence: {confidence*100:.1f}%")

import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("SENTIMENT PREDICTION SYSTEM")
reviews = [
    "amazing product love it",
    "terrible quality waste money",
    "excellent product very useful",
    "worst purchase ever",
    "really happy with this product",
    "complete garbage dont buy",
    "fantastic quality highly recommend",
    "very disappointed with this",
    "wonderful experience loved it",
    "poor quality never buy again"
]
labels = np.array([1, 0, 1, 0, 1,0, 1, 0, 1, 0])
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
reviews = [clean_text(review) for review in reviews]
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)
max_length = 6
X = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post'
)
print("\nTokenized and padded data:")
print(X)
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=max_length),
    LSTM(32),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,labels,epochs=20,batch_size=2,verbose=0)
print("\nModel training completed!")
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence,maxlen=max_length,padding='post')
    probability = model.predict(padded, verbose=0)[0][0]
    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability * 100
    else:
        sentiment = "Negative"
        confidence = (1 - probability) * 100
    return sentiment, confidence
test_sentences = [
    "This is absolutely wonderful",
    "Complete garbage dont buy",
    "I really love this product",
    "This product is terrible",
    "Amazing quality and excellent experience"
]
print("Testing new sentences:")
for text in test_sentences:
    sentiment, confidence = predict_sentiment(text)
    print(f'\nText: "{text}"')
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.1f}%")