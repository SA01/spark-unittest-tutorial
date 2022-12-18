import datetime
import unittest

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType, ArrayType
import json
import csv

from src.transformations import add_columns, running_total, group_sales_by_type


# TODO: Include testing output map and array data

class TestTransformations(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up Spark")
        conf = SparkConf().set("spark.driver.memory", "8g")

        self.spark = SparkSession \
            .builder \
            .master("local[4]") \
            .config(conf=conf) \
            .appName("test simple transformation") \
            .getOrCreate()

    def test_add_columns(self):
        # Create test data with each row as tuple
        test_data = [(1, 2), (3, 4), (5, 6)]

        # Create test DataFrame from the test data, pass the column names as required
        test_df = self.spark.createDataFrame(data=test_data, schema=["first", "second"])

        # Show data-frame
        test_df.show(truncate=False)

        # Execute transformation on the test data-frame and show the results
        result_df = test_df.transform(add_columns)
        result_df.show(truncate=False)

        # Validate column
        result_columns = result_df.columns
        self.assertIn("sum", result_columns)

        # Get rest column out of the data frame as list
        result_data = result_df.select("sum").collect()
        result_data = [item["sum"] for item in result_data]

        # Validate result column values
        self.assertListEqual(result_data, [3, 7, 11])

    def test_map_data(self):
        test_data = [
            (1, "product_1", "2022-11-01", {"store 1": 12, "store 2": 3, "online": 5}),
            (2, "product_1", "2022-11-02", {"store 1": 5, "online": 2}),
            (3, "product_1", "2022-11-04", {"store 1": 8, "store 2": 12, "online": 11}),
            (4, "product_1", "2022-11-05", {"store 1": 3, "store 2": 3})
        ]

        test_df = self.spark.createDataFrame(test_data, schema=["order_id", "product", "date", "sales"])
        test_df.show(truncate=False)
        test_df.printSchema()

        test_df_schema = StructType([
            StructField(name="order_id", dataType=IntegerType(), nullable=False),
            StructField(name="product", dataType=StringType(), nullable=False),
            StructField(name="date", dataType=StringType(), nullable=False),
            StructField(name="sales", dataType=MapType(StringType(), IntegerType(), valueContainsNull=False), nullable=False),
        ])

        test_df = self.spark.createDataFrame(test_data, schema=test_df_schema)
        test_df.show(truncate=False)
        test_df.printSchema()

    def test_list_data(self):
        test_data = [
            (1, "product_1", "2022-11-01", "2022-11-05", [3, 4, 6, 7, 12]),
            (2, "product_1", "2022-11-06", "2022-11-12", [8, 4, 3, 1, 16, 13, 25]),
            (3, "product_1", "2022-11-13", "2022-11-15", [3, 3, 6]),
            (4, "product_2", "2022-11-01", "2022-11-07", [1, 12, 6, 9, 12, 2, 2]),
        ]

        test_df_schema = StructType([
            StructField(name="order_id", dataType=IntegerType(), nullable=False),
            StructField(name="product", dataType=StringType(), nullable=False),
            StructField(name="start_date", dataType=StringType(), nullable=False),
            StructField(name="end_date", dataType=StringType(), nullable=False),
            StructField(name="sales", dataType=ArrayType(IntegerType()), nullable=False),
        ])

        test_df = self.spark.createDataFrame(test_data, schema=test_df_schema)\
            .withColumn("start_date", f.to_date("start_date"))\
            .withColumn("end_date", f.to_date("end_date"))
        test_df.show(truncate=False)
        test_df.printSchema()

        sales_data_raw = test_df.select("sales").collect()
        print(sales_data_raw)
        sales_data = [item["sales"] for item in sales_data_raw]
        print(sales_data)
        print(type(sales_data))
        print([[type(item) for item in data] for data in sales_data])

        self.assertListEqual(
            sales_data,
            [[3, 4, 6, 7, 12], [8, 4, 3, 1, 16, 13, 25], [3, 3, 6], [1, 12, 6, 9, 12, 2, 2]]
        )

    def test_group_sales_by_type(self):
        # Create test data
        test_data = [
            (1, "product_1", "online", "2022-11-01", 8),
            (2, "product_1", "online", "2022-11-02", 6),
            (3, "product_1", "online", "2022-11-04", 12),
            (4, "product_1", "retail", "2022-11-01", 11),
            (5, "product_1", "retail", "2022-11-02", 15),
            (6, "product_1", "retail", "2022-11-03", 22),
            (7, "product_1", "retail", "2022-11-04", 21),
            (8, "product_2", "online", "2022-11-02", 1),
            (9, "product_2", "online", "2022-11-03", 3),
            (10, "product_2", "retail", "2022-11-01", 1),
            (11, "product_2", "retail", "2022-11-02", 5),
            (12, "product_2", "retail", "2022-11-04", 2)
        ]

        # Define test data schema
        test_df_schema = StructType([
            StructField(name="id", dataType=IntegerType(), nullable=False),
            StructField(name="product", dataType=StringType(), nullable=False),
            StructField(name="sale_type", dataType=StringType(), nullable=False),
            StructField(name="sale_date", dataType=StringType(), nullable=False),
            StructField(name="num_sales", dataType=IntegerType(), nullable=False),
        ])

        # Create test DataFrame
        test_df = self.spark.createDataFrame(test_data, schema=test_df_schema)\
            .withColumn("sale_date", f.to_date("sale_date"))

        # Print the data frame and its schema
        test_df.show(truncate=False)
        test_df.printSchema()

        # Run the transformation on test data
        grouped_data = test_df.transform(group_sales_by_type)
        grouped_data.show(truncate=False)
        grouped_data.printSchema()

        # Collect results to validate
        validation_cols = grouped_data.select("sale_dates", "num_sales").collect()
        sale_dates = [item['sale_dates'] for item in validation_cols]
        num_sales = [item['num_sales'] for item in validation_cols]

        # Print sale_dates column result
        print(sale_dates)

        # Create and validate expected `sale_dates` result
        expected_sale_dates = [
            [
                datetime.datetime.strptime("2022-11-01", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-02", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-04", "%Y-%m-%d").date()
            ], [
                datetime.datetime.strptime("2022-11-01", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-02", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-03", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-04", "%Y-%m-%d").date()
            ],
            [
                datetime.datetime.strptime("2022-11-02", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-03", "%Y-%m-%d").date()
            ],
            [
                datetime.datetime.strptime("2022-11-01", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-02", "%Y-%m-%d").date(),
                datetime.datetime.strptime("2022-11-04", "%Y-%m-%d").date(),
            ]
        ]
        self.assertListEqual(sale_dates, expected_sale_dates)

        # Validate number of sales result
        self.assertListEqual(num_sales, [[8, 6, 12], [11, 15, 22, 21], [1, 3], [1, 5, 2]])

    def test_create_struct_data(self):
        # Create test data
        test_data = [
            (1, "product_1", "2022-11-01", {"retail": 8, "online": 12}),
            (2, "product_1", "2022-11-02", {"retail": 3}),
            (3, "product_1", "2022-11-03", {"retail": 5, "online": 2}),
            (4, "product_1", "2022-11-04", {"online": 8}),
            (5, "product_2", "2022-11-02", {"retail": 2, "online": 1}),
            (6, "product_2", "2022-11-03", {"retail": 3, "online": 2}),
        ]

        # Define test data schema
        test_df_schema = StructType([
            StructField(name="id", dataType=IntegerType(), nullable=False),
            StructField(name="product", dataType=StringType(), nullable=False),
            StructField(name="sale_date", dataType=StringType(), nullable=False),
            StructField(name="num_sales", dataType=StructType([
                StructField("retail", IntegerType(), nullable=True),
                StructField("online", IntegerType(), nullable=True),
            ]))
        ])

        # Create test DataFrame
        test_df = self.spark.createDataFrame(test_data, schema=test_df_schema) \
            .withColumn("sale_date", f.to_date("sale_date"))

        # Print the data frame and its schema
        test_df.show(truncate=False)
        test_df.printSchema()

        # method 1 - process the nested Row instances:
        num_sales = test_df.select("num_sales").collect()
        print(num_sales)

        online_sales = [item['num_sales']['online'] for item in num_sales]
        retail_sales = [item['num_sales']['retail'] for item in num_sales]

        self.assertListEqual(online_sales, [12, None, 2, 8, 1, 2])
        self.assertListEqual(retail_sales, [8, 3, 5, None, 2, 3])

        # method 2 - select to separate columns
        num_sales_method_2 = test_df.select("num_sales").select("num_sales.*").collect()
        print(num_sales_method_2)
        online_sales_method_2 = [item['online'] for item in num_sales_method_2]
        retail_sales_method_2 = [item['retail'] for item in num_sales_method_2]

        self.assertListEqual(online_sales_method_2, [12, None, 2, 8, 1, 2])
        self.assertListEqual(retail_sales_method_2, [8, 3, 5, None, 2, 3])

        # method 3 - convert the struct column to json
        num_sales_method_3 = test_df.withColumn("num_sales", f.to_json(f.col("num_sales"))).select("num_sales").collect()
        print(num_sales_method_3)

        online_sales_method_3 = [
            json.loads(item['num_sales'])['online'] if 'online' in json.loads(item['num_sales']) else None
            for item in num_sales_method_3
        ]
        retail_sales_method_3 = [
            json.loads(item['num_sales'])['retail'] if 'retail' in json.loads(item['num_sales']) else None
            for item in num_sales_method_3
        ]

        self.assertListEqual(online_sales_method_3, [12, None, 2, 8, 1, 2])
        self.assertListEqual(retail_sales_method_3, [8, 3, 5, None, 2, 3])

    def test_running_total(self):
        # # Option 1 - provide a date column
        # test_data = [
        #     (1, "product_1", datetime.strptime("2022-11-01", "%Y-%m-%d").date(), 1),
        #     (2, "product_1", datetime.strptime("2022-11-03", "%Y-%m-%d").date(), 1),
        #     (3, "product_1", datetime.strptime("2022-11-04", "%Y-%m-%d").date(), 3),
        #     (4, "product_1", datetime.strptime("2022-11-05", "%Y-%m-%d").date(), 2),
        #     (5, "product_2", datetime.strptime("2022-11-02", "%Y-%m-%d").date(), 4),
        #     (6, "product_2", datetime.strptime("2022-11-04", "%Y-%m-%d").date(), 3),
        # ]

        # Option 2 - input date as string and cast in Spark
        test_data = [
            (1, "product_1", "2022-11-01", 1),
            (2, "product_1", "2022-11-03", 1),
            (3, "product_1", "2022-11-04", 3),
            (4, "product_1", "2022-11-05", 2),
            (5, "product_2", "2022-11-02", 4),
            (6, "product_2", "2022-11-04", 3),
        ]

        test_df_columns = ["order_id", "product", "order_date", "qty"]
        test_df = self.spark.createDataFrame(test_data, test_df_columns)\
            .withColumn("order_date", f.to_date("order_date"))

        test_df.show(truncate=False)
        test_df.printSchema()

        result_df = test_df.transform(running_total)
        result_df.show(truncate=False)

        result_data = result_df.select("running_sum_qty").collect()
        result_data = [item['running_sum_qty'] for item in result_data]

        self.assertListEqual(result_data, [1, 2, 5, 7, 4, 7])

    def test_group_sales_by_type_from_file(self):
        # Define test data schema
        test_df_schema = StructType([
            StructField(name="id", dataType=IntegerType(), nullable=False),
            StructField(name="product", dataType=StringType(), nullable=False),
            StructField(name="sale_type", dataType=StringType(), nullable=False),
            StructField(name="sale_date", dataType=StringType(), nullable=False),
            StructField(name="num_sales", dataType=IntegerType(), nullable=False),
        ])

        # Read test data from .csv file
        test_df = self.spark.read.option("header", True).schema(test_df_schema).csv("test_data/test_data.csv")

        test_df.show(truncate=False)
        test_df.printSchema()

        # Perform the transformation
        result_df = test_df.transform(group_sales_by_type)
        result_df.show(truncate=False)
        result_df.printSchema()

        # Extract result data frame to list
        result_data_raw = result_df.select("num_sales").collect()
        result_data = [item["num_sales"] for item in result_data_raw]

        # Load expected data
        with open("test_data/test_result.csv", mode='r') as file_handle:
            expected_data = [json.loads(line[0]) for line in csv.reader(file_handle)]

        print(f"Expected data: {expected_data}")
        self.assertListEqual(result_data, expected_data)
