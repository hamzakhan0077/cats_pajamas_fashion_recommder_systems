import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



# articles_df = pd.read_csv("datasets/articles_sample.csv")
# transaction_df = pd.read_csv("datasets/transactions_sample.csv")
# customers_df = pd.read_csv("datasets/customers_sample.csv")
# pd.set_option("display.max_rows", None)
articles_df = pd.read_csv("datasets/articles_transactions_5.csv")
T = pd.read_csv("Datasets/transactions_5.csv")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# test_i = articles_df.sample(n=1, random_state = 1)

# test_t = pd.read_csv("Datasets/transactions_test_set.csv")
# test_u = test_t['customer_id'].values

# T = pd.read_csv("datasets/transactions_5.csv")
u = T['customer_id'].drop_duplicates().to_frame()
test_u = u.sample(n=1000, random_state = 1)
test_t = pd.DataFrame()
print("Creating Test Set")
for i,cust in enumerate(test_u['customer_id']):
    cust_transac = T[T['customer_id'] == cust]
    bottom_transac = cust_transac[-1 * round(0.20 * len(cust_transac)):]
    test_t = test_t.append(bottom_transac)
    indexs = bottom_transac.index
    T.drop(labels = indexs, axis = 0,inplace=True )
    print(i)
# test_t.to_csv("Datasets/transactions_test_set.csv")

def random_recommender(customer):
    return random.sample(sorted(articles_df['article_id'].values), 5)


def popularity_recommender(customer):
    popular_products = T['article_id'].value_counts().nlargest(5).to_frame()['article_id'].keys().values
    return popular_products


# def content_based_recommender(customer):
#     similarity = pd.DataFrame()
#     i = test_i.copy()
#     customer_trasnsactions = T[T['customer_id'] == customer]
#     # create profile
#     customer_profile = set()
#     a_set = set()
#     # Add to profile and remove the items this customer bought
#     for item in customer_trasnsactions['article_id'].drop_duplicates().values:
#         # print(i[['product_type_name','product_group_name','colour_group_name','detail_desc']][i['article_id'] == item].values[0])
#         for words in articles_df[['product_type_name','product_group_name','colour_group_name','detail_desc']][articles_df['article_id'] == item].values:
#           for word in words:
#               customer_profile.add(word)
#         i = i[i['article_id'] != item]
#
#     for test_item in i['article_id']:
#         for test_item_words in i[['product_type_name', 'product_group_name', 'colour_group_name', 'detail_desc']][i['article_id'] == test_item].values:
#             for test_word in test_item_words:
#
#                 a_set.add(test_word)
#
#         similarity = similarity.append({"customer_id":customer,
#                                         "article_id":str(test_item),
#                                         "jaccard":len(customer_profile.intersection(a_set))/len(customer_profile.union(a_set))},ignore_index=True)
#         a_set.clear()
#
#     # is always sotres so picking bottom 5 as this is in acending order
#     # print(similarity[['article_id','jaccard']][-5:].values)
#     return [int(number) for number in similarity[['article_id','jaccard']][-5:]['article_id'].values]
#     # print(similarity['article_id'][similarity['similarity'].nlargest(5)])
# # print(content_based_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644'))

# 65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644


"""
+++++++++++++++++++++++++++++++++++++++++++++++
"""
#
# i = articles_df
# # Add to profile and remove the items this customer bought
# # do this for the Candidate Items
# i['detail_desc'] = i['detail_desc'].fillna("")
# # vector TFID Matrix
# vectorizer = TfidfVectorizer(stop_words = "english")
# candidate_profile_X = vectorizer.fit_transform(i["detail_desc"])
# # Cosine Similarity
# cosine_similarity = linear_kernel(candidate_profile_X,candidate_profile_X) # to get the references to the movies
# indices = pd.Series(i.index,index=i["article_id"]).drop_duplicates()
# def rec(article_id,cosine_similarity):
#     idx = indices[article_id]
#     scores = enumerate(cosine_similarity[idx])
#     scores = sorted(scores,key=lambda val:val[1])
#     scores = scores[-6:-1]
#     # scores = scores[-2:-1] # we want the one before the Identical one (THE MOST SIMILAR)
#     # print(scores)
#     # return [i['article_id'].iloc[tar[0]] for tar  in scores]
#     return scores # now returns a list of tuple
#
#
# # print(rec(861519001,cosine_similarity))
# def content_based_recommender(customer):
#     tuple_list = [] # list with similarity and index
#     customer_purchases = T['article_id'][T['customer_id'] == customer].drop_duplicates().values
#     for product in customer_purchases:
#         tuple_list += rec(product,cosine_similarity)
#     scores = sorted(tuple_list, key=lambda val: val[1])
#     scores = scores[-6:-1] # pick bottom 5
#     return [i['article_id'].iloc[tar[0]] for tar in scores]
#


"""
+++++++++++++++++++++++++++++++++++++++++++++++++
"""











print("RUNNING PRECISION TEST")
precision_list_popularity = []
precision_list_random = []
precision_list_content = []
for j,user in enumerate(u['customer_id'].values):
    print(j)
    popular_recommendations = popularity_recommender(user)

    random_recommendations = random_recommender(user)

    # content_recommendations = content_based_recommender(user)

    u_purchases = test_t[test_t['customer_id'] == user]['article_id'].values
    precision_list_popularity.append(len(np.intersect1d(popular_recommendations,u_purchases))/5)
    precision_list_random.append(len(np.intersect1d(random_recommendations,u_purchases))/5)
    # precision_list_content.append(len(np.intersect1d(content_recommendations, u_purchases)) / 5)

print("MEAN Precison of Popularity Recommender",sum(precision_list_popularity)/len(precision_list_popularity))
print("MEAN Precison of Random Recommender",sum(precision_list_random)/len(precision_list_random))
# print("MEAN Precison of Content Based Recommender" ,sum(precision_list_content)/len(precision_list_content))
#
#


# print("RUNNING PRECISION TEST")
#
# precision_list_popularity = []
# precision_list_random = []
# precision_list_content = []
#
# popular_recommendations = popularity_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644')
#
# random_recommendations = random_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644')
#
# content_recommendations = content_based_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644')
#
# u_purchases = T[T['customer_id'] == '65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644']['article_id'].values
#
# print(u_purchases)
# print(popular_recommendations)
# print(random_recommendations)
# print(content_recommendations)
# precision_list_popularity.append(len(np.intersect1d(popular_recommendations,u_purchases))/5)
# precision_list_random.append(len(np.intersect1d(random_recommendations,u_purchases))/5)
# precision_list_content.append(len(np.intersect1d(content_recommendations, u_purchases)) / 5)
#
# print("MEAN Precison of Popularity Recommender",sum(precision_list_popularity)/len(precision_list_popularity))
# print("MEAN Precison of Random Recommender",sum(precision_list_random)/len(precision_list_random))
# print("MEAN Precison of Content Based Recommender" ,sum(precision_list_content)/len(precision_list_content))
#
#




# for customer in random_customers:
#     recommended_items = popular_items
#     for item in popular_items:
#         if item in transaction_df[['customer_id', 'article_id']][transaction_df['customer_id'] == customer][
#             'article_id'].values:
#             relevant_recommendation += 1
#     precision_per_customer.append(relevant_recommendation / len(recommended_items))
#     relevant_recommendation *= 0
# return sum(precision_per_customer) / len(precision_per_customer)
#

# transaction_df = pd.read_csv("transactions_5.csv")
# print(transaction_df)
"""
        420086     30214907  2020-08-11  ...  0.018627                 1
420087     30214908  2020-08-11  ...  0.016932                 1
420088     30214909  2020-08-11  ...  0.010153                 1
1416747    31211568  2020-09-06  ...  0.048119                 1
1416748    31211569  2020-09-06  ...  0.040102                 1
1416749    31211570  2020-09-06  ...  0.032068                 1
"""
# counter = 0
# T = transaction_df[transaction_df['t_dat'] > '2020-08-22'].copy()
# u = T['customer_id'].drop_duplicates().to_frame()
# print(u.shape)
# for cust in u['customer_id']: # will run 250619 times, Can we do it better ?
#     if T['customer_id'][T['customer_id'] == cust].count() < 5:
#         u = u[u['customer_id'] != cust].copy()
#         T = T[T['customer_id']!= cust].copy()
#     counter+=1
#     print(counter)



''' THE ONE USED
# =====================================================
# trans = pd.DataFrame()
# T = transaction_df[transaction_df['t_dat'] > '2020-08-22'].copy()
# u = T['customer_id'].value_counts()
# u = np.array(u[u >= 5].index) # 84234 Customer with more than 5 transactions
# print(len(u))
# for i,cust in enumerate(u):
#     print(i)
#     trans = trans.append(T[T['customer_id']== cust])
# trans.to_csv('transactions_5.csv')
# print(trans)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''





# 0000e9a4db2da4e9c68558ad1e7df570d02767b213ec6bcb10283ab05ee53015
# a_dict = T.groupby('customer_id').groups


# print(T[T['customer_id']== '0000e9a4db2da4e9c68558ad1e7df570d02767b213ec6bcb10283ab05ee53015'])
# print()

# T.drop(labels=a_dict['0000e9a4db2da4e9c68558ad1e7df570d02767b213ec6bcb10283ab05ee53015'], axis=0,inplace=True)
# print(T[T['customer_id']== '0000e9a4db2da4e9c68558ad1e7df570d02767b213ec6bcb10283ab05ee53015'],'**************8')




# empty_t = pd.DataFrame()
# for cust in u: # will run 145255 times, Can we do it better ?
#     empty_t.append(transaction_df[transaction_df['customer_id'] == cust])
#     counter+=1
#     print(counter)
# print(empty_t.shape)
# print(empty_t.drop_duplicates().shape)
# print(u.shape)
# my_grp_dict = T.groupby(['customer_id']).groups
# print(my_grp_dict)
# print(T.shape)
# for cust in u['customer_id']: # will run 250619 times, Can we do it better ?
#     if len(my_grp_dict[cust]) < 5:
#         T = T.drop(labels=my_grp_dict[cust], axis=0)
#         u = u[u['customer_id'] != cust].copy()
#     counter+=1
#     print(counter)
# print(T.shape())

# print(u.shape)
# for cust in u['customer_id']: # will run 250619 times, Can we do it better ?
#     if T['customer_id'][T['customer_id'] == cust].count() < 5:
#         u = u[u['customer_id'] != cust].copy()
#         T = T[T['customer_id']!= cust].copy()
#     counter+=1
#     print(counter)
# # T = T['customer_id'].count()
# print(T['customer_id'].count())
# # u = u[u['customer_id'] != cust].copy()
#
#
# # print(articles_df[articles_df['article_id']== 1080092])
# u = T['customer_id'].drop_duplicates().to_frame()

# print(u.shape)




# def data_split():
#     training_set = pd.DataFrame()
#     test_set = pd.DataFrame()
#     # Pick 1000  customers at Random
#     random_customers = customers_df.sample(n=1000, random_state = 1)
#     eighty_percent_customers , twenty_percent_customers = train_test_split(random_customers, test_size=0.2)
#
#     # Get their transactions
#     counter = 0
#
#     # For the 20 percent of the customers 80%  of their Data goes into
#     # training set and 20 percent of their data goes into the test set
#     for customer in twenty_percent_customers['customer_id'].values: # for the 20% i.e tow hundred customers
#         transactions = transaction_df[transaction_df['customer_id'] == customer] # when you query the datafrome the date is alreday soted
#         if len(transactions) > 5:
#             bottom_data = -1 * round(0.20 * len(transactions))
#             test_set = test_set.append(transactions[bottom_data:])
#             training_set = training_set.append(transactions[:bottom_data])
#         counter+=1
#         print(counter)
#     counter *= 0
#     for rest_customer in eighty_percent_customers['customer_id'].values:
#         their_transactions = transaction_df[transaction_df['customer_id'] == rest_customer]
#         training_set = training_set.append(their_transactions)
#         counter+=1
#         print(counter)
#
#     print(training_set.shape)
#     print(test_set.shape)
#
#
#
# print(data_split())