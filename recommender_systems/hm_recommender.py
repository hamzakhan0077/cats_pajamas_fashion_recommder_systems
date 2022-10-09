import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import rand
from sklearn.metrics.pairwise import euclidean_distances
import numpy.linalg as npla
from math import sqrt
from pprint import pprint


test_df = pd.read_csv("datasets/all_mix.csv")

articles_df = pd.read_csv("datasets/articles_sample.csv")

transaction_df = pd.read_csv("datasets/transactions_sample.csv")
# transaction_df = transaction_df[transaction_df['t_dat'] > '2020-08-01'].copy()
# transaction_df.to_csv('datasets/transactions_sample.csv')


customers_df = pd.read_csv("datasets/customers_sample.csv")

pd.set_option("display.max_rows", None)


def product_info():
    print(articles_df.shape)

    # print(articles_df['product_group_name'].describe(), '\n')
    # print(articles_df['product_group_name'].value_counts(dropna=False),'\n')
    #
    # print(articles_df['product_type_name'].describe(),'\n')
    # print(articles_df['product_type_name'].value_counts(dropna=False),'\n')
    #
    # print(articles_df['colour_group_name'].describe(), '\n')
    # print(articles_df['colour_group_name'].value_counts(dropna=False), '\n')
    #
    # print(articles_df['graphical_appearance_name'].describe(), '\n')
    # print(articles_df['graphical_appearance_name'].value_counts(dropna=False), '\n')

    # print(transaction_df['article_id'].value_counts())
    # print(articles_df[articles_df['article_id'] == 706016001])
    # print(articles_df[articles_df['article_id'] == 918292001])


    pass



# product_info()

def customer_info():
    print(customers_df.shape)

    # print(customers_df['Active'].describe(), '\n')
    # print(customers_df['Active'].value_counts(dropna=False), '\n')

    # print(customers_df['age'].describe(), '\n')
    # print(customers_df['age'].value_counts(dropna=False), '\n')

    # print(transaction_df[''].describe(), '\n')
    # print(transaction_df.groupby('customer_id')['price'].aggregate('sum'))
    # print(transaction_df.groupby('customer_id')['price'].aggregate('sum').describe())
    # print(transaction_df.groupby('customer_id')['price'])
    pass



# customer_info()

def transaction_info():
    print(transaction_df.shape)
    print(transaction_df['t_dat'].min(), '\n')
    print(transaction_df['t_dat'].max(), '\n')
    print(transaction_df.groupby('customer_id')['article_id'].count().describe(),'\n')
    print(transaction_df.groupby('customer_id')['price'].sum().describe(), '\n')
    print(transaction_df[['article_id','price']].max(), '\n')
    print(transaction_df[['article_id', 'price']].min(), '\n')





transaction_info()







def random_recommender(customer):
    pass



def  popularity_recommender(customer):
    pass




def data_info():
    print(transaction_df.shape)
    print(articles_df.shape)
    print(customers_df.shape)


    print(len(transaction_df['customer_id'].unique()))
    print(customers_df['customer_id'].shape)

    # for j in transaction_df['article_id'].values:
    #     print(j)

    # print(articles_df.shape)
    # print(customers_df.shape)
    # print(9475280014 in transaction_df['article_id'].values)
    # merged_df = transaction_df.merge(articles_df)
    # merged_df = merged_df[['article_id', 'product_code', 'prod_name', 'product_type_no',
    #    'product_type_name', 'product_group_name', 'graphical_appearance_no',
    #    'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
    #    'perceived_colour_value_id', 'perceived_colour_value_name',
    #    'perceived_colour_master_id', 'perceived_colour_master_name',
    #    'department_no', 'department_name', 'index_code', 'index_name',
    #    'index_group_no', 'index_group_name', 'section_no', 'section_name',
    #    'garment_group_no', 'garment_group_name', 'detail_desc']].copy().drop_duplicates()
    # merged_df.to_csv('datasets/articles_sample.csv')



    # cust_merge_df = transaction_df.merge(customers_df)
    # cust_merge_df = cust_merge_df[['customer_id', 'FN', 'Active', 'club_member_status',
    #    'fashion_news_frequency', 'age', 'postal_code']].copy().drop_duplicates()
    # cust_merge_df.to_csv('datasets/customers_sample.csv')

    # print(customers_df[customers_df['customer_id'].values in transaction_df['customer_id'].values])

# data_info()
