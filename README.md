Movie Review Sentiment Analysis with a Custom Transformer
A Natural Language Processing (NLP) project completed as part of
Udacity's Future AWS AI Programmer Nanodegree. The project builds
and trains a custom Transformer model in PyTorch to classify IMDB movie
reviews as positive or negative.

Project Overview
Sentiment analysis helps identify opinions and emotional polarity
expressed in text. In this project, a review is treated as a binary
classification problem:

Positive (1) --- the review expresses a positive opinion.
Negative (0) --- the review expresses a negative opinion.
The project follows an end-to-end machine learning workflow: load and
inspect the dataset, prepare text for modelling, implement a PyTorch
dataset and data loaders, construct a Transformer classifier, train and
validate it, evaluate it on held-out test data, and prepare functions
for predictions on new reviews.

Important architecture note: The model is trained from scratch.
The bert-base-uncased Hugging Face tokenizer is used to convert text
into token IDs; pretrained BERT model weights are not used.

Results
The submitted notebook's saved training and evaluation outputs reported
the following results:

Metric Result

Best epoch 5
Best validation accuracy 75.96%
Held-out test accuracy 75.86%
Required test-accuracy threshold Greater than 75%
Difference above threshold 0.86 percentage points

The model exceeded the project's stated test-accuracy target. The test
set contains 25,000 reviews and is used for final evaluation after the
best checkpoint is selected using validation accuracy.

Training history
Epoch   Training loss   Training accuracy   Validation accuracy
    1          0.6409              61.88%                69.40%
    2          0.5324              72.90%                72.68%
    3          0.4720              77.24%                73.72%
    4          0.4260              80.36%                74.12%
    5          0.3860              82.80%                75.96%
Training loss decreased each epoch, while training and validation
accuracy improved across the five epochs. Training accuracy is higher
than validation accuracy at the final epoch, so further experiments
could examine generalization and possible overfitting. The reported test
result is a single evaluation from the saved best-validation checkpoint.

Dataset
The project uses the Stanford IMDB movie review dataset.

25,000 labelled training reviews: 12,500 positive and 12,500
negative.
25,000 labelled test reviews: 12,500 positive and 12,500 negative.
The 25,000-review training portion is split into:
22,500 training examples (90% of the original training portion)
2,500 validation examples (10% of the original training portion)
The class distribution is balanced in the train, validation, and
test splits.
Dataset resources: - Stanford IMDB dataset
page - Dataset
archive

The notebook expects the extracted dataset to be available in an
aclImdb directory, with train/pos, train/neg, test/pos, and
test/neg subdirectories.

Workflow and Implementation
1. Load, inspect, and explore data
The notebook loads the labelled reviews into Pandas DataFrames and
checks the dataset dimensions, labels, and class balance. It calculates
review character and word counts and uses visualizations to explore the
text data.

2. Tokenize the reviews
The Hugging Face bert-base-uncased tokenizer converts raw text into
vocabulary token IDs. Each review is truncated or padded to a fixed
length of 128 tokens so that batches have consistent tensor
dimensions.

3. Build a PyTorch input pipeline
A custom IMDBDataset class returns token IDs and their sentiment
labels. PyTorch DataLoader objects provide shuffled training batches
and non-shuffled validation and test batches, using a batch size of
32.

4. Build the custom Transformer classifier
The DemoGPT model is assembled from PyTorch components rather than
loading a pretrained Transformer model. Its main configuration is:

Component Configuration

Token vocabulary bert-base-uncased tokenizer
vocabulary (30,522 tokens in the
recorded run)

Maximum sequence length 128 tokens

Token embedding dimension 128

Transformer blocks 4

Attention heads per block 4

Dimension per attention head 32

Feed-forward hidden dimension 512

Dropout 0.1

Output classes 2 (negative and positive)
The model adds token embeddings and positional embeddings, processes the
sequence through Transformer blocks containing multi-head attention and
feed-forward layers, applies layer normalization, mean-pools the token
representations, and sends the resulting review representation to a
linear classification head that returns two logits.

5. Train and select the best checkpoint
The model is trained for 5 epochs with:

Loss: cross-entropy loss
Optimizer: AdamW
Learning rate: 2e-4
Weight decay: 0.01
Gradient clipping: maximum norm of 1.0
Checkpoint selection: highest validation accuracy
The best model checkpoint is saved as sentimentscope_best.pt. The
checkpoint stores the model state, configuration, maximum sequence
length, best validation accuracy, and best epoch.

6. Evaluate and predict
The notebook calculates accuracy by comparing predicted classes with the
true labels. After training, it reloads the best-validation checkpoint
and evaluates it once on the held-out test set. Helper functions are
also provided to load the saved model and tokenizer and to predict
sentiment and class probabilities for new reviews.

Tech Stack
Python
PyTorch
Hugging Face Transformers tokenizer
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn (for dataset splitting, where used)
Jupyter Notebook / Google Colab
CUDA-enabled GPU for training acceleration when available
How to Run
Requirements
Use Python in a Jupyter Notebook or Google Colab environment. A
CUDA-enabled GPU is recommended because the custom Transformer is
trained from scratch.

Install the main dependencies if they are not already available:

pip install torch transformers pandas numpy matplotlib seaborn scikit-learn tqdm
Steps
Clone or download this repository and open the project notebook.
Download the Stanford IMDB
dataset if it is
not already supplied in your runtime.
Extract it so the notebook can access the aclImdb directory with
the expected train/test positive/negative folders.
Select a GPU runtime if one is available.
Run the notebook cells from top to bottom.
Confirm the training history and final test-evaluation outputs in
the notebook.
The best checkpoint, sentimentscope_best.pt, is created during
training. Keep it available if you want to reload the trained model
for predictions.
The saved accuracy values in this README describe the recorded run
included in the submitted notebook. If you change the model,
hyperparameters, data, or random seed and run training again, your
results may differ.

Example: Predict Sentiment for New Reviews
After running the notebook and creating the checkpoint, use its helper
functions:

trained_model, trained_tokenizer, saved_config, saved_length = load_sentimentscope()

predictions = predict_sentiment_batch(
    [
        "I loved the acting and the story.",
        "The plot was dull and poorly written."
    ],
    trained_model,
    trained_tokenizer,
    saved_config,
    saved_length
)

print(predictions)
The prediction helper returns a sentiment label (Positive or
Negative) and the model's negative and positive class probabilities
for each review. These example reviews illustrate how to call the
helper; the exact predictions should be confirmed by running the
notebook.

Key Learnings
Preparing and validating a real-world text dataset before model
training.
Exploring sentiment labels and review-length distributions using
descriptive statistics and visualizations.
Creating a custom PyTorch Dataset and DataLoader pipeline for
tokenized text.
Implementing Transformer building blocks, including attention heads,
multi-head attention, feed-forward layers, normalization, and
dropout.
Training a neural text classifier and monitoring loss, training
accuracy, and validation accuracy.
Selecting a model checkpoint using validation performance and
measuring generalization on held-out test data.
Structuring a reproducible notebook workflow and documenting
measured results accurately.
Limitations and Future Improvements
The model is a relatively small Transformer trained from scratch; it
does not use pretrained BERT weights.
The final test accuracy is only modestly above the required 75%
threshold.
Training accuracy exceeds validation accuracy, so additional
experiments could help assess and improve generalization.
Future experiments could compare additional epochs, learning rates,
sequence lengths, model depth, embedding size, regularization, and
random seeds.
A pretrained BERT classifier could be explored as a separate
comparison, but it would be a different modelling approach from this
custom Transformer implementation.
Project Context
This project was completed for Udacity's Future AWS AI Programmer
Nanodegree as a practical exercise in NLP, PyTorch model construction,
Transformer architecture, training, validation, and evaluation.

Acknowledgements
Udacity --- learning programme and
project context.
Stanford IMDB
dataset ---
movie-review sentiment data.
Hugging Face
Transformers ---
tokenizer utilities.
PyTorch --- model implementation and
training framework.
Writing code? Meet Codex in the desktop app
Pair with Codex in your terminal or IDE, free for a limited time in your ChatGPT plan.




