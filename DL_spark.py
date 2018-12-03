# Databricks notebook source




import os
SUBMIT_ARGS = "--packages databricks:spark-deep-learning:1.0.0-spark2.3-s_2.11 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DL with Spark Deep Cognition").getOrCreate()
sc = spark.sparkContext

# COMMAND ----------

sc = spark.sparkContext
sc

# COMMAND ----------

# load image

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables"))

# COMMAND ----------

from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
# Read images using Spark ... as a DataFrame.
# Each image is stored as a row in the imageSchema format.
image_cle = ImageSchema.readImages("dbfs:/FileStore/tables/cle/").withColumn("label", lit(0))
image_cre = ImageSchema.readImages("dbfs:/FileStore/tables/cre/").withColumn("label", lit(1))
image_ole = ImageSchema.readImages("dbfs:/FileStore/tables/ole/").withColumn("label", lit(2))
image_ore = ImageSchema.readImages("dbfs:/FileStore/tables/ore/").withColumn("label", lit(3))

# COMMAND ----------

image_cle.show(),image_cre.show(),image_ole.show(),image_ore.show()

# COMMAND ----------

type(image_cle)

# COMMAND ----------

# Create training & test DataFrames for transfer learning - this piece of code is longer than transfer learning itself below!
re_df = image_ore.unionAll(image_cre)
le_df= image_ole.unionAll(image_cle)

re_train, re_test = re_df.randomSplit([0.6, 0.4])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
le_train, le_test= le_df.randomSplit([0.6, 0.4])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)

train_df = re_train.unionAll(le_train)
test_df = re_test.unionAll(le_test)

# Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# This ensure that each of the paritions has a small size.
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

# COMMAND ----------

#re_df.show(),le_df.show()
train_df.show(),test_df.show()

# COMMAND ----------

train_df.count(),test_df.count()

# COMMAND ----------

#image_ore.count(),image_cle.count(),re_df.count(),re_train.count(),re_test.count()

# COMMAND ----------

# Transfer Learning technique
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

# COMMAND ----------




# COMMAND ----------

p_model = p.fit(train_df)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

tested_df = p_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))

# COMMAND ----------

tested_df.show()

# COMMAND ----------

from sparkdl import DeepImagePredictor

# COMMAND ----------

image_df = ImageSchema.readImages("/FileStore/tables/salma3.jpg")

# COMMAND ----------

predictions_df = p_model.transform(image_df)

# COMMAND ----------

predictions_df.show()

# COMMAND ----------

predictions_df.select("prediction").show(truncate=False,n=3)

# COMMAND ----------

x=predictions_df

# COMMAND ----------

dddd = x.toPandas()
dddd

# COMMAND ----------

y=dddd['prediction'][0]

# COMMAND ----------

y = int(y)
type(y)

# COMMAND ----------

def class_predict(y):
  if y==0:
    return('close left eye')
  elif y==1:
    return('close right eye')
  elif y==2:
    return('open left eye')
  elif y==3:
     return('open right eye')
  else:
    return ('error')

# COMMAND ----------

class_predict(y)

# COMMAND ----------


