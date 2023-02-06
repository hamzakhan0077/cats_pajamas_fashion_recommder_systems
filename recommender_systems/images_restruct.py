# import pandas as pd
# import numpy as np
# import shutil

# import os

# articles_df = pd.read_csv("datasets/articles.csv")






# print(articles_df.shape)
# print(articles_df.iloc[0])

# 108775015
# 0108775015

# print(articles_df['product_type_name'][articles_df['article_id'] == 108775015])

# x = open("classes.txt","a")
# print(x.write(str(articles_df['product_type_name'].drop_duplicates().tolist()))) #131 classes

"""  DOG Wear APpears twice in the loop, when the program create
the directory for it the second time it crashes saying that it already exist"""
#
# i = 0
# remaining_classes = []
# # for target_class in articles_df['product_type_name'].drop_duplicates().values:
# for target_class in  ['Eyeglasses', 'Wireless earphone case', 'Stain remover spray', 'Clothing mist']:
#     os.mkdir(f'datasets/images/{target_class.replace("/","")}')
#     for j in articles_df['article_id'][articles_df['product_type_name'] == f'{target_class}']:
#         j = str(j)
#         try:
#             shutil.copy(f'datasets/images/0{j[:2]}/0{j}.jpg', f'datasets/images/{target_class}/')
#         except:
#             continue
#     i+=1
#     print(i)
#
#
