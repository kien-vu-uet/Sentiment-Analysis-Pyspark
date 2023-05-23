# Sentiment-Analysis-Pyspark

Dataset: Amazon book reviews
-- Retrieval from https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

# Setup
## Installation

**Note**:
Note:
The current version is only compatible with python>=3.6
```bash
git clone https://github.com/kien-vu-uet/Sentiment-Analysis-Pyspark.git
cd Sentiment-Analysis-Pyspark
pip install -r requirements.txt
```

## Quick start training: 
The device needs to be pre-installed with Hadoop >= 3.3.1 and Spark >= 2.0

## Step 1: Prepare dataset

Experimental example based on Amazon-book-reviews data pair of 900 comments

Data includes reviews (`review/text`) and scores (`review/score`):

| Data set               | Reviews    |                    Download                   |
| :--------------------: | :--------: | :-------------------------------------------: |
| Amazon-book-reviews    | 900,000    | via Kaggle in above link                      |

Push data to hdfs: 
```bash
hdfs dfs -copyFromLocal /path/to/local/data /path/to/hadoop
```

## Step 2: Training models

Select model and train:
```bash
python tfidf.py # For TFIDF extractor and classifier (NB, LR, SVM)
python tfidf_evaluate.py # For evaluating F1 score 
# or
python word2vec.py # For Word2vec extractor and classifier (LR, SVM)
python word2vec_evaluate.py # For evaluating F1 score
# or
# First, run script in ngram_capture.ipynb
python ngram_tfidf.py # For Ngram extractor and classifier (NB, LR, SVM)
python ngram_evaluate.py # For evaluating F1 score
```
## Step 3: Infer
```bash
   python infer.py --extractor=/your/choose --classifer=/your/choose --input=/path/to/your/txt/input --output=/path/to/your/txt/output
```
