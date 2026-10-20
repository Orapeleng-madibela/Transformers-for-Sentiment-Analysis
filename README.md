# Transformers-for-Sentiment-Analysis: Movie Review Sentiment Analysis with Transformers

## Udacity AWS AI Programmer Project

**An end-to-end Natural Language Processing project using PyTorch and a custom Transformer architecture to classify movie reviews as positive or negative.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/Field-NLP-purple.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Dataset](https://img.shields.io/badge/Dataset-IMDB-yellow.svg)](https://ai.stanford.edu/~amaas/data/sentiment/)
[![Status](https://img.shields.io/badge/Project-Research%20%26%20Learning-informational.svg)]()

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Transformer Architecture](#transformer-architecture)
- [Model Configuration](#model-configuration)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges and Lessons Learned](#challenges-and-lessons-learned)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)
- [Applications](#applications)
- [Author](#author)
- [References](#references)

---

## Project Overview

**SentimentScope** is a Natural Language Processing (NLP) and deep learning project that explores how Transformer neural networks can be used to understand text and classify sentiment.

The project uses the Stanford Large Movie Review Dataset, which contains labelled movie reviews. A custom Transformer classifier is trained to predict whether a review expresses positive or negative sentiment.

Rather than relying on a pretrained BERT classification model, this implementation builds the Transformer architecture from PyTorch components and trains its model weights from scratch. The Hugging Face `bert-base-uncased` tokenizer is used to convert review text into token IDs.

The project covers an end-to-end machine learning workflow, including data acquisition, exploratory data analysis, text preprocessing, model construction, training, validation, checkpointing, testing, and inference.

**Project repository:**  
https://github.com/Orapeleng-madibela/Transformers-for-Sentiment-Analysis

## Problem Statement

Online reviews contain valuable information about customer opinions, product experiences, and satisfaction. However, manually analysing large volumes of text can be time-consuming and difficult to scale.

This project investigates how a deep learning model can automatically classify movie reviews into two sentiment categories:

- **Positive:** The review expresses a favourable opinion.
- **Negative:** The review expresses an unfavourable opinion.

The objective is to build and evaluate a Transformer-based text classifier and understand the practical challenges involved in developing an NLP model.

## Project Objectives

The main objectives are to:

1. Load and validate a real-world labelled text dataset.
2. Explore sentiment distributions and review characteristics.
3. Prepare raw text for neural network processing.
4. Tokenise reviews using a pretrained tokenizer vocabulary.
5. Implement a custom Transformer architecture using PyTorch.
6. Train the model using supervised learning.
7. Monitor validation performance and preserve the best checkpoint.
8. Evaluate generalisation on a held-out test dataset.
9. Use the trained model to make predictions on new movie reviews.
10. Document the implementation so that other developers can reproduce and extend the experiment.

## Key Features

- **End-to-end NLP workflow:** From raw reviews to model predictions.
- **Exploratory data analysis:** Investigates dataset composition and review characteristics.
- **Text tokenisation:** Converts natural language into numerical sequences.
- **Custom Transformer:** Implements attention, feed-forward networks, and Transformer blocks.
- **Supervised model training:** Uses labelled positive and negative reviews.
- **Validation monitoring:** Tracks model performance during training.
- **Best-model checkpointing:** Saves the checkpoint associated with the strongest validation performance.
- **Held-out testing:** Evaluates the selected model on a separate test dataset.
- **Inference support:** Provides functionality for classifying new review text.
- **Reproducibility:** Sets random seeds and documents the model configuration.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network implementation and training |
| Hugging Face Transformers | Tokenizer loading and text tokenisation |
| pandas | Dataset organisation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| scikit-learn | Stratified training-validation split and evaluation utilities |
| Jupyter Notebook | Interactive development, experimentation, and documentation |
| Git and GitHub | Version control and project portfolio |

## Dataset

This project uses the **Stanford Large Movie Review Dataset**, commonly known as the IMDB sentiment dataset.

**Official dataset resources:**

- [Stanford dataset page](https://ai.stanford.edu/~amaas/data/sentiment/index.html)
- [Download the dataset archive](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

The dataset contains 50,000 labelled movie reviews, divided into 25,000 training reviews and 25,000 test reviews. It also includes additional unlabelled reviews that are not required for this project's supervised training workflow.

| Dataset component | Number of reviews |
|---|---:|
| Labelled training set | 25,000 |
| Labelled test set | 25,000 |
| Additional unlabelled reviews | 50,000 |
| Total including unlabelled reviews | 100,000 |

The labelled dataset is balanced between positive and negative reviews.

### Sentiment labels

| Label | Meaning |
|---|---|
| `0` | Negative |
| `1` | Positive |

### Dataset preparation

The notebook:

1. Downloads and extracts the official dataset if the expected folder structure is not already available.
2. Reads positive and negative reviews from the corresponding folders.
3. Associates each review with its sentiment label.
4. Organises the labelled reviews into data structures suitable for analysis and training.
5. Separates the official training data into training and validation subsets.
6. Reserves the official test set for final evaluation.

The training-validation split uses 90% of the original labelled training set for model training and 10% for validation. This produces 22,500 training reviews and 2,500 validation reviews when the full dataset is available.

The official test set remains separate from training and validation to provide a more independent estimate of model performance.

## Project Workflow

The project follows a structured machine learning pipeline.

1. **Data acquisition:** Obtain the official IMDB dataset.
2. **Data validation:** Check that the expected dataset folders and labelled reviews are available.
3. **Exploratory data analysis:** Examine class balance, review lengths, and other descriptive statistics.
4. **Data preparation:** Organise reviews and sentiment labels.
5. **Tokenisation:** Convert text into token IDs using the `bert-base-uncased` tokenizer.
6. **Sequence preparation:** Apply the configured maximum sequence length and prepare batches for model input.
7. **Model construction:** Build a custom Transformer-based binary classifier.
8. **Training:** Optimise model parameters using labelled training examples.
9. **Validation:** Measure performance after each training epoch and identify the best checkpoint.
10. **Testing:** Evaluate the selected model on the held-out test set.
11. **Inference:** Apply the trained model to new movie reviews.
12. **Reporting:** Record the actual evaluation results and identify possible improvements.

## Transformer Architecture

The model is implemented using custom PyTorch components.

### 1. Tokenisation and embeddings

The `bert-base-uncased` tokenizer converts raw review text into token IDs using its vocabulary and subword tokenisation rules.

**Important:** The pretrained tokenizer is used for tokenisation only. This project does not load pretrained BERT classification weights. The neural network's trainable parameters are initialised and trained as part of this custom model.

The model converts token IDs into learned embeddings and incorporates positional information so that the network can process the sequence of tokens.

### 2. Attention heads

An attention head learns relationships between token representations. Attention allows the model to combine information from different positions in a review rather than processing each word in isolation.

### 3. Multi-head attention

Multi-head attention combines multiple attention heads, allowing the network to learn different relationships between tokens.

### 4. Feed-forward network

A feed-forward network transforms token representations after attention has been applied.

### 5. Transformer blocks

The model stacks Transformer blocks containing attention and feed-forward operations, with residual connections, normalisation, and dropout as implemented in the notebook.

### 6. Classification head

The final representation is aggregated and passed to a classification layer that produces two output logits:

- Negative sentiment
- Positive sentiment

The logits are used with the cross-entropy loss function during training. Softmax can be applied to the output logits to obtain class probabilities during inference.

### Architecture summary

`Review text → Tokenizer → Token IDs → Embeddings + Positional Information → Transformer Blocks → Representation Pooling → Classification Head → Sentiment Prediction`

## Model Configuration

The notebook uses the following configuration.

| Parameter | Configuration |
|---|---|
| Model architecture | Custom Transformer classifier |
| Framework | PyTorch |
| Tokenizer | `bert-base-uncased` |
| Transformer layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Maximum sequence length | 128 tokens |
| Dropout | 0.1 |
| Batch size | 32 |
| Optimiser | AdamW |
| Loss function | Cross-entropy |
| Gradient clipping | Enabled |
| Training epochs | 5 |
| Training-validation split | 90% / 10% of labelled training data |
| Number of output classes | 2 |
| Random seed | 42 |

These are the notebook's configured values. The final performance depends on successfully running training and evaluation.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- Git.
- VS Code with the Python and Jupyter extensions, or another Jupyter-compatible environment.
- An internet connection for the initial dataset and tokenizer downloads.
- Sufficient memory and storage for the dataset and model.
- A CUDA-compatible GPU is recommended for faster training.

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Orapeleng-madibela/Transformers-for-Sentiment-Analysis.git
cd Transformers-for-Sentiment-Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Otherwise, install the notebook dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn torch transformers tqdm jupyter
```

PyTorch installation requirements can differ depending on whether you intend to use a CPU or a supported GPU. Consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if you need a specific CUDA-enabled build.

## Running the Project

### Option A: Visual Studio Code

1. Open the cloned repository folder in VS Code.
2. Install the Python and Jupyter extensions if they are not already installed.
3. Open the project's `.ipynb` notebook.
4. Select the virtual environment's Python interpreter and notebook kernel.
5. Run the notebook cells from top to bottom, or select **Run All**.
6. Allow the notebook to download and extract the dataset if necessary.
7. Wait for training and evaluation to finish.
8. Save the notebook with its outputs if you want reviewers to see the execution results.

### Option B: Jupyter Notebook

Start Jupyter from the activated virtual environment:

```bash
jupyter notebook
```

Open the project notebook in the browser and execute the cells in order.

### Dataset download

The notebook is designed to download the official Stanford dataset automatically when it cannot find the expected `aclImdb` directory.

If you download the dataset manually, extract it and ensure the notebook points to the directory containing the `train` and `test` folders.

### Training considerations

Training a Transformer from scratch can take a significant amount of time on a CPU. A supported GPU is recommended.

For a quick debugging run, you can temporarily reduce the number of training epochs or use a smaller subset of the training data. Restore the intended configuration before collecting final results.

## Model Evaluation

The model should be evaluated on data that was not used to optimise its weights.

### Validation

Validation is performed during training to monitor generalisation and select the best checkpoint. The validation results can help identify whether the model is improving or beginning to overfit.

### Held-out test evaluation

The selected checkpoint is evaluated against the official test set after training. Test performance should be reported using the actual results printed by the notebook.

### Evaluation metrics

The principal metric is classification accuracy:

`Accuracy = Correct predictions / Total predictions`

Additional useful metrics for a more comprehensive evaluation include:

- **Precision:** The proportion of predicted positive examples that are actually positive.
- **Recall:** The proportion of actual positive examples correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** The number of correct and incorrect predictions for each class.

Accuracy alone may not explain all classification errors. Precision, recall, F1-score, and a confusion matrix would provide a more detailed assessment if added to the evaluation pipeline.

## Making Predictions

The notebook includes inference functionality for predicting the sentiment of new text after the model has been trained.

Example reviews that can be used to test the inference function:

```python
sample_reviews = [
    "An excellent movie with outstanding performances and a powerful story.",
    "The plot was boring, the acting was weak, and the movie was disappointing.",
    "The film had some interesting moments, but the overall experience was average.",
]
```

Pass these reviews to the notebook's prediction function after training and loading the best checkpoint. The model returns a positive or negative class prediction.

The third example is intentionally mixed in sentiment. Binary classifiers can still assign it to one of the two classes even when the review is ambiguous.

**Note:** The sample inputs above illustrate how to test inference. Their predicted labels and confidence scores are not precomputed results.

## Results

The notebook must be executed before numerical results can be reported. The values below are deliberately left for measured outputs rather than estimated or invented figures.

| Evaluation item | Result |
|---|---|
| Best validation accuracy | To be measured |
| Held-out test accuracy | To be measured |
| Test precision | To be measured if calculated |
| Test recall | To be measured if calculated |
| Test F1-score | To be measured if calculated |
| Training duration | To be measured |

After running the notebook:

1. Record the best validation accuracy.
2. Record the final held-out test accuracy.
3. Add other metrics only if the notebook actually calculates them.
4. Include labelled plots of training and validation loss or accuracy if available.
5. Discuss the errors and limitations revealed by the results.

Do not claim that the model meets a particular accuracy threshold until the test result has been verified.

## Project Structure

The exact files in the GitHub repository may vary. The intended organisation is:

```text
Transformers-for-Sentiment-Analysis/
│
├── SentimentScope_Final_Notebook.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                       # Optional local dataset directory
│
├── checkpoints/                # Optional model checkpoint directory
│
└── results/                    # Optional plots and evaluation outputs
```

The notebook contains the primary implementation, including data preparation, model construction, training, evaluation, and inference.

The `data/`, `checkpoints/`, and `results/` directories are suggested organisational conventions; they need not exist unless you create them. Large datasets and generated model checkpoints generally should not be committed directly to GitHub.

## Challenges and Lessons Learned

This project provides practical experience with several important aspects of machine learning development.

### Working with text data

Raw reviews must be transformed into consistent numerical inputs before a neural network can process them. Tokenisation and sequence-length handling affect what information reaches the model.

### Implementing attention

Building attention and Transformer blocks provides a deeper understanding of how the architecture combines contextual information across a sequence.

### Training a model from scratch

Training a custom model requires attention to parameter initialisation, optimisation, memory use, batch processing, and training duration.

### Separating training, validation, and testing

Using validation data for checkpoint selection and keeping the test set separate helps provide a more credible estimate of generalisation.

### Reproducibility and evaluation

Documenting configuration, setting random seeds, saving checkpoints, and recording measured metrics are essential for making experiments easier to reproduce and assess.

## Limitations

- The classifier predicts only positive or negative sentiment.
- Reviews are limited to the configured maximum sequence length, so information outside that limit may be omitted.
- The custom Transformer is trained from scratch rather than benefiting from pretrained BERT model weights.
- Training time and model performance depend on hardware, hyperparameters, and execution.
- Results on movie reviews do not guarantee similar performance on social media posts, product reviews, or other domains.
- Sarcasm, mixed opinions, negation, and context-dependent language can make sentiment classification difficult.
- Performance claims require completed training and evaluation outputs.

## Future Improvements

Potential next steps include:

- Add precision, recall, F1-score, and a confusion matrix.
- Compare the custom Transformer with a TF-IDF plus Logistic Regression baseline.
- Compare against a pretrained DistilBERT or BERT classifier.
- Tune learning rate, embedding dimension, number of layers, and attention heads.
- Experiment with longer sequence lengths and different pooling strategies.
- Add error analysis for misclassified reviews.
- Implement early stopping and learning-rate scheduling.
- Save a reproducible experiment configuration and training logs.
- Add automated tests for preprocessing, tensor shapes, checkpoint loading, and inference.
- Develop a lightweight API or web interface for interactive sentiment predictions.
- Explore deployment, monitoring, and performance considerations for real-world use.

## Skills Demonstrated

This project is relevant to entry-level Data Science, Machine Learning, AI, and Python development applications.

**Programming and data handling**
- Python programming
- Data loading and validation
- Data manipulation with pandas and NumPy
- Exploratory data analysis and visualisation

**Machine learning and NLP**
- Text classification
- Text tokenisation and sequence preparation
- Supervised learning
- Training-validation-test workflows
- Model evaluation and generalisation

**Deep learning**
- PyTorch
- Neural network construction
- Attention mechanisms and Transformer blocks
- Loss functions and optimisation
- Gradient clipping and checkpointing
- Inference using a trained model

**Engineering practices**
- Reproducible experiments
- Dependency management
- Git and GitHub
- Technical documentation
- Identifying limitations and planning improvements

## Applications

The techniques explored in this project can inform systems such as:

- Customer feedback analysis
- Product and service review classification
- Brand sentiment monitoring
- Feedback triage and prioritisation
- Text analytics dashboards
- Automated review summarisation pipelines when combined with suitable additional models

The current project is an experimental binary classifier, not a production sentiment-analysis service. Applying it to other domains would require additional testing and potentially domain-specific training data.

## Author

**Orapeleng Timothy Madibela**

Interested in data analysis, data science, machine learning, artificial intelligence, and software development.

- **GitHub:** https://github.com/Orapeleng-madibela
- **LinkedIn:** https://www.linkedin.com/in/orapeleng-madibela-0074b824a/
- **Portfolio:** https://orapeleng-madibela.github.io/my-portfolio/

For collaboration, feedback, or professional opportunities, please connect through GitHub or LinkedIn.

## References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies. [Read the paper](https://aclanthology.org/P11-1015/).
2. Stanford AI Lab. [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/index.html).
3. PyTorch. [Official documentation](https://pytorch.org/docs/stable/index.html).
4. Hugging Face. [Transformers documentation](https://huggingface.co/docs/transformers/index).
5. Vaswani, A., et al. (2017). *Attention Is All You Need.* [Read the paper](https://arxiv.org/abs/1706.03762).

---

If you find this project useful, consider starring the repository or opening an issue with suggestions for improvements.
