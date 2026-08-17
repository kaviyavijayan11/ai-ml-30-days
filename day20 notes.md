Day 20 - Week 3 Self Test 
Q1 - Dense Layer
===

Connects every neuron from the previous layer to every neuron in the current layer.
Learns patterns using weights, bias, and activation functions.

## Q2 - Dropout(0.5)

Training time: randomly disables 50% of neurons temporarily to reduce overfitting.
Prediction time: all neurons are active.

## Q3 - Conv2D vs MaxPooling2D

Conv2D → detects features (edges, patterns) using filters
MaxPooling2D → keeps important features, reduces spatial size and computation

## Q4 - Train vs Test Accuracy Gap

100% train accuracy, 60% test accuracy → Overfitting
Solution: more data, dropout, L2 regularization, early stopping, simpler model

## Q5 - Tokenizer + Padding

Tokenizer: converts words → numbers
Padding: makes all sentences the same length (required for fixed neural net input size)

## Q6 - Embedding Layer

One-hot encoding only identifies a word (sparse, no relationships).
Embedding learns a compact numerical representation that captures relationships between words (e.g. similar meaning words end up closer).

## Q7 - LSTM vs Bidirectional LSTM

LSTM → one direction context (past → future only)
BiLSTM → forward + backward context (uses full sentence, better for context-heavy tasks)

## Q8 - 100% Training Accuracy but Fails on New Sentences

Dataset was small + model complexity was high, so the model memorized training sentences
instead of learning generalizable patterns. Same words/structure it saw → works.
New phrasing → fails, since it never learned to generalize.

## Q9 - Early Stopping

Monitors validation loss during training to check if it's still improving.
patience = number of epochs to wait without improvement before stopping.
Prevents overfitting by stopping before validation loss starts rising.

## Q10 - L2 Regularization

Penalizes large weights, encouraging smaller weights → simpler decision boundary → reduces overfitting.

