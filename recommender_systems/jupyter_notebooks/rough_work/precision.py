import pandas as pd
import numpy as np
from numpy.random import rand
import random

from sklearn.model_selection import train_test_split
articles_df = pd.read_csv("datasets/articles_sample.csv")
transaction_df = pd.read_csv("datasets/transactions_sample.csv")
customers_df = pd.read_csv("datasets/customers_sample.csv")
pd.set_option("display.max_rows", None)



# The data of 287377 customers in the training set
# The 20 precent data of 71845 customers in  test set and remain 80 percent of their data in training set

# trans_train_df = pd.DataFrame()
# trans_test_df = pd.DataFrame()
# cust_training_set, cust_test_set = train_test_split(customers_df, test_size=0.001)
#
#
# for customer in cust_test_set['customer_id']: # the remain 20 percent of customers
#     # the rest of their transactions i.e eighty percent goes into trainig set
#     transactions = transaction_df[transaction_df['customer_id'] == customer]
#     # print(transactions['article_id'].count())
#     if len(transactions['article_id'].values) > 5:
#         eighty_percent , twenty_percent = train_test_split(transactions, test_size=0.2)
#         trans_test_df = trans_test_df.append(twenty_percent)
#         cust_training_set = cust_training_set.append(eighty_percent)

#         # total_bottom_data = -1 * (round(0.20 * len(transactions['article_id'].values)))
#         # bottom_data = transactions['article_id'].values[total_bottom_data:] # picking up bottom 20 percent of data
#


# print(trans_train_df.shape)
# print(trans_test_df.shape)
#

    #trans_train,trans_test = train_test_split(transaction_df[transaction_df['customer_id'] == customer], test_size=0.2)
    # Works for 1 ,
    # Problem could be -- there might be a customer with 1 purchase
    # how would 1 be splatted into 20 and 80

    # trans_train_df = trans_train_df.append(trans_train)
    # trans_test_df = trans_test_df.append(trans_test)

#############################################################################
# Split the transactions data frame rather than customers

def trans_split():
    trans_train_df = pd.DataFrame()
    trans_test_df = pd.DataFrame()
    all_transactions = transaction_df.copy()
    grouped_trans = all_transactions['customer_id'].drop_duplicates()
    cust_training_set, cust_test_set = train_test_split(grouped_trans, test_size=0.0001)

    print(cust_training_set.shape)
    print(cust_test_set.shape)
    # exit()
    for customer in cust_test_set.values:  # the remain 20 percent of customers
    # the rest of their transactions i.e eighty percent goes into trainig set
        transactions = all_transactions[all_transactions['customer_id'] == customer]
        # print(transactions['article_id'].count())
        if transactions['article_id'].count() > 5:
            eighty_percent , twenty_percent = train_test_split(transactions, test_size=0.2)
            trans_test_df = trans_test_df.append(twenty_percent)
            trans_train_df = trans_train_df.append(eighty_percent)
            trans_train_df = trans_train_df.append(cust_training_set)


    print(trans_test_df.shape)
    print(trans_test_df.shape)





    # print(transactions['article_id'].count())

trans_split()




def random_recommender(customer):
    return random.sample(sorted(articles_df['article_id'].values), 5)


def popularity_recommender(customer):
    popular_products = transaction_df['article_id'].value_counts().nlargest(5).to_frame()['article_id'].keys().values
    return popular_products


def random_recommender_evaluation():
    # Randomly choose 100 users
    # take 20 % of data per user eg if a user has 100 transactions we take 20
    # for each customer recommend them items
    # check how many of those recommendation are actually in their purchases

    random_customers = np.random.choice(customers_df['customer_id'].values, 10)
    relevant_recommendation = 0
    precision_per_customer = []
    for customer in random_customers:
        recommended_items = random_recommender(customer)
        for item in recommended_items:
            if item in transaction_df[['customer_id', 'article_id']][transaction_df['customer_id'] == customer][
                'article_id'].values:
                relevant_recommendation += 1
        precision_per_customer.append(relevant_recommendation / len(recommended_items))
        relevant_recommendation *= 0
    return sum(precision_per_customer) / len(precision_per_customer)


def popularity_recommender_evaluation():
    random_customers = customers_df['customer_id'].values[-10:]
    print(len(random_customers))
    # random_customers = ['06b253c1f2946179fdef2598b59e9df4bf01d6be4f015ab6ae2d8dd5723d2745']
    popular_items = popularity_recommender(
        popularity_recommender(random_customers[0]))  # it will  recommend same items to every customer
    relevant_recommendation = 0
    precision_per_customer = []
    # print(706016001 in transaction_df[['customer_id', 'article_id']][transaction_df['customer_id'] == random_customers[0]][
    #             'article_id'].values)

    for customer in random_customers:
        recommended_items = popular_items
        for item in popular_items:
            if item in transaction_df[['customer_id', 'article_id']][transaction_df['customer_id'] == customer][
                'article_id'].values:
                relevant_recommendation += 1
        precision_per_customer.append(relevant_recommendation / len(recommended_items))
        relevant_recommendation *= 0
    return sum(precision_per_customer) / len(precision_per_customer)
