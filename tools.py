# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import configparser
import datetime as dt

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql import SQLContext
from pyspark.sql.functions import isnan, when, count, col, udf, dayofmonth, dayofweek, month, year, weekofyear
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *

import plotly.plotly as py
import plotly.graph_objs as go
import requests
requests.packages.urllib3.disable_warnings()


def visualize_missing_values(df):
    """Given a dataframe df, visualize it's missing values by columns
    :param df:
    :return:
    """
    # lets explore missing values per column
    nulls_df = pd.DataFrame(data= df.isnull().sum(), columns=['values'])
    nulls_df = nulls_df.reset_index()
    nulls_df.columns = ['cols', 'values']

    # calculate % missing values
    nulls_df['% missing values'] = 100*nulls_df['values']/df.shape[0]

    plt.rcdefaults()
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x="cols", y="% missing values", data=nulls_df)
    ax.set_ylim(0, 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()



def visualize_missing_values_spark(df):
    """Visualize missing values in a spark dataframe
    
    :param df: spark dataframe
    """
    # create a dataframe with missing values count per column
    nan_count_df = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
    
    # convert dataframe from wide format to long format
    nan_count_df = pd.melt(nan_count_df, var_name='cols', value_name='values')
    
    # count total records in df
    total = df.count()
    
    # now lets add % missing values column
    nan_count_df['% missing values'] = 100*nan_count_df['values']/total
    
    plt.rcdefaults()
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x="cols", y="% missing values", data=nan_count_df)
    ax.set_ylim(0, 100)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
       
def aggregate_temperature_data(df):
    """Aggregate clean temperature data at country level
    
    :param df: spark dataframe of clean global temperaturs data
    :return: spark dataframe consisting of countries average temperatures
    """
    new_df = df.select(['Country', 'AverageTemperature']).groupby('Country').avg()
    
    new_df = new_df.withColumnRenamed('avg(AverageTemperature)', 'average_temperature')
    
    return new_df