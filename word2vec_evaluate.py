
# Export java11 to use
import os
os.environ['JAVA_HOME'] = '/home/nlplab/.jdk/jdk-11.0.19+7'


# os.system('hdfs dfs -copyFromLocal /home/team1/notebooks/amazon-books-reviews/part2_900k.csv /user/team1')


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


# Create a spark session
spark = SparkSession.builder \
    .appName("SentimentAnalysisWord2vec") \
    .master("local[*]") \
    .config("spark.driver.memory", "100g") \
    .config("spark.executor.memory", "100g") \
    .config("spark.memory.offHeap.enabled","true") \
    .config("spark.memory.offHeap.size","100g") \
    .getOrCreate()


# Load the sentiment data
# Assume the data has two columns: body and score
# Score is an integer from 1 to 5
print('READ DATASET...')
data = spark.read.csv('test.csv', inferSchema=True, header=True, multiLine=True, quote='"', escape='"')
data = data.select('review/score', (lower(regexp_replace('review/text', "[^a-zA-Z\\s]", "")).alias('review/text')))
data = data.dropna()


# Convert to 2 label 0, 1
data = data.replace(1, 0, subset=["review/score"])
data = data.replace(2, 0, subset=["review/score"])
data = data.replace(3, 0, subset=["review/score"])
data = data.replace(4, 1, subset=["review/score"])
data = data.replace(5, 1, subset=["review/score"])


# Define the pipeline that includes tokenize, hashingTF and IDF
print('PREPROCESSING...')
# Tokenize text
preprocess = PipelineModel.load('pipelines/preprocess/tokenizer_stopwordremover')
# preprocess = preprocess.fit(data)
data = preprocess.transform(data)


# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
data = data.withColumn("words_stemmed", stemmer_udf("words")).select('words_stemmed', 'review/score')

# Filter length word > 3
filter_length_udf = udf(lambda row: [x for x in row if len(x) >= 3], ArrayType(StringType()))
data = data.withColumn('words', filter_length_udf(col('words_stemmed')))


# Split the data into train and test sets
train, test = data.randomSplit([0.9, 0.1], seed=42)

model = PipelineModel.load('./pipelines/word2vec_lr_svm')

# Predict on the test set
predictions = model.transform(test)


# Evaluate the model performance
print("EVALUATION...")
evaluator = MulticlassClassificationEvaluator(labelCol="review/score", \
                    predictionCol="LR_prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print(f"The f1 of the model Logistic Regression is {f1:0.2f}")


# Evaluate the model performance
evaluator = MulticlassClassificationEvaluator(labelCol="review/score", \
                    predictionCol="SVM_prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print(f"The f1 of the model Linear SVM is {f1:0.2f}")


spark.stop()