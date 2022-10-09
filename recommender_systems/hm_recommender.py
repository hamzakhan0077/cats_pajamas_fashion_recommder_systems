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

articles_df = pd.read_csv("datasets/articles.csv")
transaction_df = pd.read_csv("datasets/transactions_train.csv")
customers_df = pd.read_csv("datasets/customers.csv")


def test():
    pass

def data_info():
    # print(articles_df.describe())
    pass




data_info()
test()