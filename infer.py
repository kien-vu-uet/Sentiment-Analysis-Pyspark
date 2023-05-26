# Export java11 to use
import os
os.environ['JAVA_HOME'] = '/home/team1/.jdk/jdk-11.0.19+7'
os.environ["SPARK_HOME"] = "/opt/spark"

# import findspark and initialize it
import findspark
findspark.init("/opt/spark")

import argparse


# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC, MultilayerPerceptronClassifier
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel

from nltk.corpus import stopwords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main argument parser")
    parser.add_argument("--extractor", choices=("tfidf", "word2vec", "ngram"), \
                        help="Choose Feature Extractor for serving")
    parser.add_argument("--classifier", choices=("lr", "nb", "svm"), \
                        help="Choose Classifier for serving")
    parser.add_argument("--input", type=str, required=True, default='input', \
                        help="Input file")
    parser.add_argument("--output", type=str, required=True, default='output', \
                        help="Output file")

    args = parser.parse_args()

    # Create a spark session
    spark = SparkSession.builder \
        .appName("SentimentAnalysisWord2vec") \
        .master("local[*]") \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "100g") \
        .config("spark.memory.offHeap.enabled","true") \
        .config("spark.memory.offHeap.size","100g") \
        .getOrCreate()
    
    file_inp = args.input
    file_out = args.output
    file = open(file_inp, 'r')
    reviews = file.readlines()
    file.close()
    
    data = spark.createDataFrame([reviews]).toDF("review/text")

    # Define the pipeline that includes tokenize, hashingTF and IDF
    print('PREPROCESSING...')
    # Tokenize text
    preprocess = PipelineModel.load('pipelines/preprocess/tokenizer_stopwordremover')
    # preprocess = preprocess.fit(data)
    data = preprocess.transform(data)

    print('APPLYING PIPELINE...')
    if args.extractor == 'tfidf':
        extractor = PipelineModel.load('pipelines/tfidf_nb_lr_svm')
        data = extractor.transform(data)
    elif args.extractor == 'word2vec':
        # Stem text
        stemmer = SnowballStemmer(language='english')
        stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
        data = data.withColumn("words_stemmed", stemmer_udf("words"))

        # Filter length word > 3
        filter_length_udf = udf(lambda row: [x for x in row if len(x) >= 3], ArrayType(StringType()))
        data = data.withColumn('words', filter_length_udf(col('words_stemmed')))

        extractor = PipelineModel.load('./pipelines/word2vec_lr_svm')
        data = extractor.transform(data)
    else:
        ngram_1 = PipelineModel.load('pipelines/preprocess/1-gram_idf')
        ngram_2 = PipelineModel.load('pipelines/preprocess/2-gram_idf')
        extractor = PipelineModel.load('pipelines/ngram_tfidf_nb_lr_svm')

        data = ngram_1.transform(data)
        data = ngram_2.transform(data)
        data = extractor.transform(data)

    if args.extractor == 'word2vec' and args.classifier == 'nb':
        raise Exception("Naive Bayes cannot work with Word2Vec extractor!")
    else:
        if args.classifier == 'nb': 
            my_list = [row['NB_prediction'] for row in data.select('NB_prediction').collect()]
        elif args.classifier == 'lr':
            my_list = [row['LR_prediction'] for row in data.select('LR_prediction').collect()]
        else:
            my_list = [row['SVM_prediction'] for row in data.select('SVM_prediction').collect()]

    # print(my_list)
    # open a file for writing
    f = open(file_out, "w")
    # create an array
    # write each element to the file
    print(*my_list, file=f)
    # close the file
    f.close()

    spark.stop()
    
