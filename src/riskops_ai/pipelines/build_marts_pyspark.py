"""Optional PySpark pipeline skeleton.

The main project uses DuckDB so it runs easily on any laptop. This file exists so you can
show PySpark/Databricks-style thinking and extend it later if Java/Spark is installed.
"""
from __future__ import annotations

from riskops_ai.config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, ensure_dirs


def build_with_pyspark() -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -r requirements-optional.txt") from exc

    ensure_dirs()
    spark = (
        SparkSession.builder
        .appName("riskops-ai-local")
        .master("local[*]")
        .getOrCreate()
    )

    apps = spark.read.option("header", True).option("inferSchema", True).csv(str(BRONZE_DIR / "loan_applications.csv"))
    customers = spark.read.option("header", True).option("inferSchema", True).csv(str(BRONZE_DIR / "customers.csv"))

    approval_funnel = (
        apps.join(customers.select("customer_id", "employment_type", "credit_segment"), "customer_id", "left")
        .withColumn("application_month", F.date_trunc("month", F.col("application_date")))
        .groupBy("application_month", "product", "channel", "employment_type", "credit_segment")
        .agg(
            F.count("*").alias("applications"),
            F.sum(F.when(F.col("approval_status") == "Approved", 1).otherwise(0)).alias("approved_applications"),
            F.round(100 * F.avg(F.when(F.col("approval_status") == "Approved", 1).otherwise(0)), 2).alias("approval_rate_pct"),
            F.round(F.avg("risk_score"), 2).alias("avg_risk_score"),
        )
    )

    approval_funnel.write.mode("overwrite").parquet(str(GOLD_DIR / "spark_gold_approval_funnel.parquet"))
    spark.stop()


if __name__ == "__main__":
    build_with_pyspark()
