import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import rand
from sklearn.metrics.pairwise import euclidean_distances
import numpy.linalg as npla
from math import sqrt
from pprint import pprint
import random
from sklearn.model_selection import train_test_split
test_df = pd.read_csv("datasets/all_mix.csv")
articles_df = pd.read_csv("datasets/articles_sample.csv")
transaction_df = pd.read_csv("datasets/transactions_sample.csv")

# transaction_df = transaction_df[transaction_df['t_dat'] > '2020-08-01'].copy()
# transaction_df.to_csv('datasets/transactions_sample.csv')


customers_df = pd.read_csv("datasets/customers_sample.csv")

pd.set_option("display.max_rows", None)


def product_info():
    # print(articles_df.shape)
    # print(articles_df.columns)
    # print(articles_df['prod_name'].describe(), '\n')
    # print(articles_df['prod_name'].value_counts(dropna=False), '\n')
    #
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
    #
    # print(transaction_df['article_id'].value_counts())
    # print(articles_df[articles_df['article_id'] == 706016001])
    # print(articles_df[articles_df['article_id'] == 918292001])

    df = transaction_df.merge(articles_df ,left_on='article_id', right_on='article_id')
    print(df.shape)



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
    # print(transaction_df.groupby('customer_id')['price'].aggregate('sum').nlargest(3))
    # print(transaction_df['customer_id'][transaction_df['article_id']==706016001 ].values[0])




# customer_info()

def transaction_info():
    print(transaction_df.shape)
    print(transaction_df['t_dat'].min(), '\n')
    print(transaction_df['t_dat'].max(), '\n')
    print(transaction_df.groupby('customer_id')['article_id'].count().describe(),'\n')
    print(transaction_df.groupby('customer_id')['price'].sum().describe(), '\n')
    print(transaction_df[['article_id','price']].max(), '\n')
    print(transaction_df[['article_id', 'price']].min(), '\n')






# transaction_info()





customers = customers_df.to_records()[-5:]



def random_recommender(customer):
    # print(f"{customer.customer_id} ", np.random.choice(articles_df['article_id'].values,5))
    return random.sample(sorted(articles_df['article_id'].values),5)


# print(random_recommender(customers[0]))
# for customer in customers:
#     random_recommender(customer)
#
#


def  popularity_recommender(customer):
    # Top 5 most bought
    popular_products = transaction_df['article_id'].value_counts().nlargest(5).to_frame()['article_id'].keys().values
    return popular_products
    # for product in popular_products:
    #     print(product)
        # print(articles_df[['prod_name','product_group_name','colour_group_name']][articles_df['article_id'] == product],'\n')
        # print(articles_df['article_id'][articles_df['article_id'] == product].values,'\n')

# print(popularity_recommender(customers[1]))




def data_info():
    # print(transaction_df.shape)
    # print(articles_df.shape)
    # print(customers_df.shape)
    #
    #
    # print(len(transaction_df['customer_id'].unique()))
    # print(customers_df['customer_id'].shape)
    #
    # print(len(transaction_df['article_id'].unique()))
    # print(articles_df['article_id'].count())

    # for j in transaction_df['article_id'].values:
    #     print(j)

    # print(articles_df.shape)
    # print(customers_df.shape)
    # print(9475280014 in transaction_df['article_id'].values)
    T = pd.read_csv("datasets/transactions_5.csv")
    A = pd.read_csv("datasets/articles.csv")
    # merged_df = transaction_df.merge(articles_df)
    merged_df = T.merge(A)
    merged_df = merged_df[['article_id', 'product_code', 'prod_name', 'product_type_no',
       'product_type_name', 'product_group_name', 'graphical_appearance_no',
       'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
       'perceived_colour_value_id', 'perceived_colour_value_name',
       'perceived_colour_master_id', 'perceived_colour_master_name',
       'department_no', 'department_name', 'index_code', 'index_name',
       'index_group_no', 'index_group_name', 'section_no', 'section_name',
       'garment_group_no', 'garment_group_name', 'detail_desc']].copy().drop_duplicates()
    merged_df.to_csv('datasets/articles_transactions_5.csv')



    # cust_merge_df = transaction_df.merge(customers_df)
    # cust_merge_df = cust_merge_df[['customer_id', 'FN', 'Active', 'club_member_status',
    #    'fashion_news_frequency', 'age', 'postal_code']].copy().drop_duplicates()
    # cust_merge_df.to_csv('datasets/customers_sample.csv')

    # print(customers_df[customers_df['customer_id'].values in transaction_df['customer_id'].values])

# data_info()



def random_recommender_evaluation():
    # Randomly choose 100 users
    # take 20 % of data per user eg if a user has 100 transactions we take 20
    # for each customer recommend them items
    # check how many of those recommendation are actually in their purchases

    random_customers = np.random.choice(customers_df['customer_id'].values,10)
    relevant_recommendation = 0
    precision_per_customer = []
    for customer in random_customers :
        recommended_items = random_recommender(customer)
        for item in recommended_items:
            if item in transaction_df[['customer_id','article_id']][transaction_df['customer_id'] == customer]['article_id'].values:
                relevant_recommendation += 1
        precision_per_customer.append(relevant_recommendation/len(recommended_items))
        relevant_recommendation *= 0
    return sum(precision_per_customer)/len(precision_per_customer)




def popularity_recommender_evaluation():
    random_customers = customers_df['customer_id'].values[-10:]
    print(len(random_customers))
    # random_customers = ['06b253c1f2946179fdef2598b59e9df4bf01d6be4f015ab6ae2d8dd5723d2745']
    popular_items = popularity_recommender(popularity_recommender(random_customers[0])) # it will  recommend same items to every customer
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





# print(random_recommender_evaluation())

# print(popularity_recommender_evaluation())


def sparsity():
    out = {} # (product , Number of times purchased)
    sparsity_of_each_product = {}
    cust_sample = customers_df.sample(frac=0.001, replace=True, random_state=1)
    product_types = articles_df['product_type_name'].drop_duplicates().count() # 34293 - Numpy Array
     # 34293
    print(customers_df.shape)
    print(cust_sample.shape)
    product_list = []
    # All the customer ID
    for customer in cust_sample['customer_id'].values: # Complexity - O(Number of custoemrs)
        # Get their purchases out of THEIR transactions
        for purchased in transaction_df[transaction_df['customer_id'] == customer]['article_id'].drop_duplicates().values:
            # see what did they purchases
            product = articles_df['product_type_name'][articles_df['article_id'] == purchased].values[0]
            # add the item if it is not arleady there - eg 2 trousers will count as 1
            if product not in product_list:
                product_list.append(product)

        # then for every product that they bought calculate the sparsity
        for unique_product in product_list:
            if unique_product in out.keys():
                out[unique_product]+=1
            else:
                out[unique_product] = 1
        product_list.clear()
    for p_type in out.keys():
        sparsity = 1 - out[p_type]/ customers_df['customer_id'].count() * product_types
        sparsity_of_each_product[p_type] = sparsity
    print(len(sparsity_of_each_product))
    print(articles_df['product_type_name'].drop_duplicates().count())
    return sparsity_of_each_product

# print(sparsity()) # takes 30 sec for 359 Custs
# estimated 5 mins for 3500 customers`
# for 359222 customers the estimated computation time is 500 Minutes - 8.3 hrs
# can we make it faster ?

# This should work because they way we split our initial data, Eveyr product has one customer
# this meaans since the abouve code computes sparsity for each customer
# they every pproducts, sparsity will be computer


