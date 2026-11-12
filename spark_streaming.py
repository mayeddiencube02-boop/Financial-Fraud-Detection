"""
Step 6b (Spark) - PySpark Structured Streaming fraud scoring.

Three modes, run in this order:
  python spark_streaming.py prepare   -> writes spark_train.csv and 10 small files in streaming_input/
  python spark_streaming.py train     -> trains a Spark MLlib logistic regression, saves fraud_model_spark/
  python spark_streaming.py stream    -> watches streaming_input/ and scores new files as they arrive

Needs: Java 17 installed (JAVA_HOME set) and  python -m pip install pyspark
Note: PySpark may not support the very newest Python versions. If install or startup fails,
create a venv with Python 3.11 or 3.12 and run this file there.
"""
import os
import sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
STREAM_DIR = os.path.join(PROJECT_DIR, "streaming_input")
MODEL_DIR = os.path.join(PROJECT_DIR, "fraud_model_spark")
TRAIN_CSV = os.path.join(PROJECT_DIR, "spark_train.csv")
ALERTS_CSV = os.path.join(PROJECT_DIR, "spark_alerts.csv")
THRESH = 0.70      # alert threshold on the Spark model's probability (adjust after looking at results)

NUM = ["Transaction_Amount", "Average_Spend", "amount_to_avg_ratio", "Previous_Transactions",
       "Account_Age_Days", "hour", "Is_International", "suspicious_keyword_flag",
       "is_night", "risk_flag_count"]
CAT = ["Location", "Device_Type", "Merchant_Category", "Payment_Method"]
KEEP = ["Transaction_ID", "Customer_ID"] + NUM + CAT + ["Fraudulent"]


def prepare():
    df = pd.read_csv(os.path.join(PROJECT_DIR, "features_clean.csv"), parse_dates=["Transaction_Date"])
    train, test = train_test_split(df, test_size=0.20, stratify=df["Fraudulent"], random_state=42)
    train[KEEP].to_csv(TRAIN_CSV, index=False)
    test = test.sort_values("Transaction_Date")
    if os.path.exists(STREAM_DIR):
        shutil.rmtree(STREAM_DIR)
    os.makedirs(STREAM_DIR)
    for i in range(10):
        chunk = test.iloc[i * 100:(i + 1) * 100][KEEP]
        chunk.to_csv(os.path.join(STREAM_DIR, f"batch_{i + 1:02d}.csv"), index=False)
    print(f"Wrote {TRAIN_CSV} ({len(train)} rows) and 10 files in {STREAM_DIR}")


def get_spark():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.appName("FraudDetectionSystem")
            .master("local[*]").config("spark.sql.shuffle.partitions", "4").getOrCreate())


def train():
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import functions as F

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    data = spark.read.csv(TRAIN_CSV, header=True, inferSchema=True)
    ratio = data.filter("Fraudulent = 0").count() / data.filter("Fraudulent = 1").count()
    data = data.withColumn("weight", F.when(F.col("Fraudulent") == 1, ratio).otherwise(1.0))

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in CAT]
    encoder = OneHotEncoder(inputCols=[c + "_idx" for c in CAT],
                            outputCols=[c + "_ohe" for c in CAT], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=NUM + [c + "_ohe" for c in CAT], outputCol="raw_features")
    from pyspark.ml.feature import StandardScaler
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol="Fraudulent",
                            weightCol="weight", maxIter=100)
    model = Pipeline(stages=indexers + [encoder, assembler, scaler, lr]).fit(data)
    model.write().overwrite().save(MODEL_DIR)
    print("Model saved to", MODEL_DIR)

    # sanity check on the held-out stream files
    test = spark.read.csv(STREAM_DIR, header=True, inferSchema=True)
    pred = model.transform(test)
    ev = BinaryClassificationEvaluator(labelCol="Fraudulent", rawPredictionCol="rawPrediction")
    print("Held-out ROC-AUC:", round(ev.setMetricName("areaUnderROC").evaluate(pred), 4))
    print("Held-out PR-AUC :", round(ev.setMetricName("areaUnderPR").evaluate(pred), 4))
    spark.stop()


def stream():
    from pyspark.ml import PipelineModel
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StringType, DoubleType

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    schema = StructType()
    for c in KEEP:
        schema = schema.add(c, StringType() if c in ["Transaction_ID", "Customer_ID"] + CAT else DoubleType())

    model = PipelineModel.load(MODEL_DIR)
    transactions = (spark.readStream.schema(schema)
                    .option("header", True).option("maxFilesPerTrigger", 1)
                    .csv(STREAM_DIR))

    scored = (model.transform(transactions)
              .withColumn("fraud_prob", vector_to_array("probability")[1])
              .select("Transaction_ID", "Customer_ID", "Transaction_Amount", "Location",
                      "fraud_prob", "Fraudulent"))

    def process_batch(batch_df, batch_id):
        pdf = batch_df.toPandas()
        if pdf.empty:
            return
        high = pdf[pdf["fraud_prob"] >= THRESH]
        caught = int(high["Fraudulent"].sum())
        print(f"[batch {batch_id}] {len(pdf)} transactions | {len(high)} HIGH-risk alerts | "
              f"{caught}/{int(pdf['Fraudulent'].sum())} real frauds caught")
        for _, r in high.head(5).iterrows():
            print(f"   ALERT! {r['Transaction_ID']} | {r['Customer_ID']} | "
                  f"amount {r['Transaction_Amount']:.2f} | {r['Location']} | risk {r['fraud_prob']:.0%}")
        high.to_csv(ALERTS_CSV, mode="a", index=False, header=not os.path.exists(ALERTS_CSV))

    if os.path.exists(ALERTS_CSV):
        os.remove(ALERTS_CSV)
    query = scored.writeStream.foreachBatch(process_batch).outputMode("append").start()
    print("Streaming... (stops by itself after 60 seconds)")
    query.awaitTermination(60)
    query.stop()
    spark.stop()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"prepare": prepare, "train": train, "stream": stream}.get(mode, lambda: print(__doc__))()
