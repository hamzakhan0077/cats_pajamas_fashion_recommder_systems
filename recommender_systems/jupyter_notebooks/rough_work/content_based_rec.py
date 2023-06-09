import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# [811773002, 811773001, 811773002, 839899002, 841473001]

articles_df = pd.read_csv("datasets/articles_transactions_5.csv")
T = pd.read_csv("Datasets/transactions_5.csv")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

test_t = pd.read_csv("Datasets/transactions_test_set.csv")
# test_i = articles_df.sample(n=500, random_state = 1)
u = T['customer_id'].drop_duplicates().to_frame()

vectorizer = TfidfVectorizer(stop_words = "english")

# vectorizer = CountVectorizer(binary=True,stop_words = "english")
# for binary include the last one as wel, (  [-6:])

# [755876005, 859399006, 814817002, 814817001, 838900001]

# similarity = pd.DataFrame()
i = articles_df
# Add to profile and remove the items this customer bought
# do this for the Candidate Items
i['detail_desc'] = i['detail_desc'].fillna("")
# vector TFID Matrix
candidate_profile_X = vectorizer.fit_transform(i["detail_desc"])
# Cosine Similarity
cosine_similarity = linear_kernel(candidate_profile_X,candidate_profile_X) # to get the references to the movies
indices = pd.Series(i.index,index=i["article_id"]).drop_duplicates()
def rec(article_id,cosine_similarity):
    idx = indices[article_id]
    scores = enumerate(cosine_similarity[idx])
    scores = sorted(scores,key=lambda val:val[1])
    scores = scores[-6:-1]
    # scores = scores[-2:-1] # we want the one before the Identical one (THE MOST SIMILAR)
    # print(scores)
    # return [i['article_id'].iloc[tar[0]] for tar  in scores]
    return scores # now returns a list of tuple


# print(rec(861519001,cosine_similarity))
def content_based_recommender(customer):
    tuple_list = [] # list with similarity and index
    customer_purchases = T['article_id'][T['customer_id'] == customer].drop_duplicates().values
    for product in customer_purchases:
        tuple_list += rec(product,cosine_similarity)
    tuple_list = set(tuple_list)
    tuple_list = list(tuple_list)
    scores = sorted(tuple_list, key=lambda val: val[1])
    print(scores)
    scores = scores[-6:-1] # pick bottom 5
    print(scores)
    return [i['article_id'].iloc[tar[0]] for tar in scores]





print(content_based_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644'))

#
# def content_based_recommender(customer):
#     vectorizer = TfidfVectorizer(stop_words = "english")
#
#     similarity = pd.DataFrame()
#     i = test_i.copy()
#     customer_transaction = T[T['customer_id'] == customer]
#
#     # create profile
#     customer_profile = ""
#     a_set = set()
#     # Add to profile and remove the items this customer bought
#     # do this for the Candidate Items
#     i['detail_desc'] = i['detail_desc'].fillna("")
#
#     # vector TFID Matrix
#     candidate_profile_X = vectorizer.fit_transform(i["detail_desc"])
#
#     # Cosine Similarity
#     cosine_simlarity = linear_kernel(candidate_profile_X,candidate_profile_X) # to get the references to the movies
#
#     indices = pd.Series(i.index,index=i["article_id"]).drop_duplicates()
#
#     # print(customer_transaction['article_id'].drop_duplicates().values)
#     # print(cosine_simlarity)
#     for item in customer_transaction['article_id'].drop_duplicates().values:
#         print(indices)
    #     idx = indices[item]
    #     for i in cosine_simlarity[idx]:
    #         print(i)
    # #
    #     break


        # customer_profile += " " + articles_df['detail_desc'][articles_df['article_id'] == item].values[0]
        # i = i[i['article_id'] != item]
    # cust_profile_X = vectorizer.fit_transform([customer_profile])
    # print(vectorizer.get_feature_names_out() )
    # print(cust_profile_X.toarray())








        # for words in articles_df[['product_type_name', 'product_group_name', 'colour_group_name', 'detail_desc']][articles_df['article_id'] == item].values:



    # for item in customer_trasnsactions['article_id'].drop_duplicates().values:
    #     # print(i[['product_type_name','product_group_name','colour_group_name','detail_desc']][i['article_id'] == item].values[0])
    #     for words in articles_df[['product_type_name','product_group_name','colour_group_name','detail_desc']][articles_df['article_id'] == item].values:
    #       for word in words:
    #           customer_profile.add(word)
    #     i = i[i['article_id'] != item]
    #
    # for test_item in i['article_id']:
    #     for test_item_words in i[['product_type_name', 'product_group_name', 'colour_group_name', 'detail_desc']][i['article_id'] == test_item].values:
    #         for test_word in test_item_words:
    #
    #             a_set.add(test_word)
    #
    #     similarity = similarity.append({"customer_id":customer,
    #                                     "article_id":str(test_item),
    #                                     "jaccard":len(customer_profile.intersection(a_set))/len(customer_profile.union(a_set))},ignore_index=True)
    #     a_set.clear()
    #
    # # is always sotres so picking bottom 5 as this is in acending order
    # # print(similarity[['article_id','jaccard']][-5:].values)
    # return [int(number) for number in similarity[['article_id','jaccard']][-5:]['article_id'].values]
    # print(similarity['article_id'][similarity['similarity'].nlargest(5)])
# print(content_based_recommender('65cb62c794232651e2ac711faa11c2b4e3d41d5f3b59b50bee3ffde1d5776644'))




















