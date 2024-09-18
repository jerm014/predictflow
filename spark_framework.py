#!/usr/bin/env python3
"""Module for spark framework in datapipeline"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

def create_spark_session():
    """Create a Spark session with the necessary dependencies"""
    return SparkSession.builder \
        .appName("PredictFlow") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
        .getOrCreate()

def read_from_kafka(spark):
    """Read data from Kafka and return a streaming DataFrame"""
    kafka_bootstrap_servers = "localhost:9092"
    kafka_topic = "predictflow_data"
    return spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .load()

def process_data(df):
    """Process data for schema and return results"""
    schema = StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("data", StringType(), True)
    ])

    return df.select(
                     from_json(
                               col("value").cast("string"),
                               schema).alias("data")) \
        .select("data.*")

def write_to_postgres(df, epoch_id, username, password):
    """Write processed data to PostgreSQL database"""
    df.write \
        .format("jdbc") \
        .option("url", "jdbc:postgresql://localhost:5432/predictflow") \
        .option("dbtable", "processed_data") \
        .option("user", username) \
        .option("password", password) \
        .mode("append") \
        .save()

if __name__ == "__main__":
    spark = create_spark_session()
    streaming_df = read_from_kafka(spark)
    processed_df = process_data(streaming_df)

    query = processed_df \
        .writeStream \
        .foreachBatch(write_to_postgres) \
        .outputMode("update") \
        .start()

    query.awaitTermination()
