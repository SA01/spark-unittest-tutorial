from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as f


def add_columns(df: DataFrame) -> DataFrame:
    return df.withColumn("sum", f.col("first") + f.col("second"))


def running_total(df: DataFrame) -> DataFrame:
    running_window = Window()\
        .partitionBy("product")\
        .orderBy(f.col("order_date").asc())

    running_total_result = df.withColumn("running_sum_qty", f.sum("qty").over(running_window))
    return running_total_result


def group_sales_by_type(df: DataFrame) -> DataFrame:
    grouped_sales_result = df\
        .groupBy("product", "sale_type")\
        .agg(
            f.collect_list("sale_date").alias("sale_dates"),
            f.collect_list("num_sales").alias("num_sales")
        )

    return grouped_sales_result
