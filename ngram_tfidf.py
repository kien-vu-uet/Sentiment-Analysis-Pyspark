# Export java11 to use
import os
os.environ['JAVA_HOME'] = '/home/team1/.jdk/jdk-11.0.19+7'
os.environ["SPARK_HOME"] = "/opt/spark"

# import findspark and initialize it
import findspark
findspark.init("/opt/spark")

# Import the necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import udf, col, lower, regexp_replace, when
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover, NGram, CountVectorizer, VectorAssembler
from pyspark.ml.classification import NaiveBayes,LogisticRegression, LinearSVC, MultilayerPerceptronClassifier
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel

# Create a spark session
spark = SparkSession.builder \
    .appName("SentimentAnalysisTFIDF") \
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
data = spark.read.csv('part2_900k.csv', inferSchema=True, header=True, multiLine=True, quote='"', escape='"')
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
# Tokenize text
preprocess = PipelineModel.load('pipelines/preprocess/tokenizer_stopwordremover')
# preprocess = preprocess.fit(data)
data = preprocess.transform(data)

ngram_1 = PipelineModel.load('pipelines/preprocess/1-gram_idf')
ngram_2 = PipelineModel.load('pipelines/preprocess/2-gram_idf')

data = ngram_1.transform(data)
data = ngram_2.transform(data)

# Split the data into train and test sets
train, test = data.randomSplit([0.9, 0.1], seed=42)

assembler = [VectorAssembler(
    inputCols=['1_tfidf', '2_tfidf'],
    outputCol="features"
)]

clf = [
    NaiveBayes(featuresCol="features", labelCol="review/score", 
                smoothing=1.0, modelType='multinomial', 
                predictionCol='NB_prediction', rawPredictionCol='NB_rawPredictionCol',
                probabilityCol='NB_probability'),

    LogisticRegression(featuresCol="features", labelCol="review/score", 
                        maxIter=1000, regParam=0.01, elasticNetParam=0.01, 
                        predictionCol='LR_prediction', rawPredictionCol='LR_rawPredictionCol',
                        probabilityCol='LR_probability'),

    LinearSVC(featuresCol="features", labelCol="review/score", 
                regParam=0.01, maxIter=1000,
                predictionCol='SVM_prediction', rawPredictionCol='SVM_rawPredictionCol')
]

#Add to pipeline
pipeline = Pipeline(stages = assembler + clf)

# Fit the model on the train set
print('START TRAINING...')
model = pipeline.fit(train)

#Save pipeline
model.save('pipelines/ngram_tfidf_nb_lr_svm')

# Predict on the test set
predictions = model.transform(test)

# Evaluate the model performance
print("EVALUATION...")
nb_evaluator = MulticlassClassificationEvaluator(labelCol="review/score", predictionCol="NB_prediction", metricName="accuracy")
nb_accuracy = nb_evaluator.evaluate(predictions)
print(f"The accuracy of the model Naive Bayes is {nb_accuracy:0.2f}")

nb_evaluator.setMetricName('f1')
nb_f1 = nb_evaluator.evaluate(predictions)
print(f"The f1 of the model Naive Bayes is {nb_f1:0.2f}")

lr_evaluator = MulticlassClassificationEvaluator(labelCol="review/score", predictionCol="LR_prediction", metricName="accuracy")
lr_accuracy = lr_evaluator.evaluate(predictions)
print(f"The accuracy of the model Logistic Regression is {lr_accuracy:0.2f}")

lr_evaluator.setMetricName('f1')
lr_f1 = lr_evaluator.evaluate(predictions)
print(f"The f1 of the model Logistic Regression is {nb_f1:0.2f}")

svm_evaluator = MulticlassClassificationEvaluator(labelCol="review/score", predictionCol="SVM_prediction", metricName="accuracy")
svm_accuracy = svm_evaluator.evaluate(predictions)
print(f"The accuracy of the model Linear SVM is {svm_accuracy:0.2f}")

svm_evaluator.setMetricName('f1')
svm_f1 = svm_evaluator.evaluate(predictions)
print(f"The f1 of the model Linear SVM is {svm_f1:0.02f}")

spark.stop()