import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import rand
from sklearn.metrics.pairwise import euclidean_distances
import numpy.linalg as npla
from math import sqrt
from pprint import pprint

mv_df = pd.read_csv("datasets/movies.csv")
rt_df = pd.read_csv("datasets/ratings.csv")

def data():
	print(mv_df.columns)
	print(mv_df.shape)
	print(mv_df.dtypes)
	print(mv_df.describe(include="all"))

def playing_around():
	# features = ["genres"]
	# X = mv_df[features].values
	# print(X)

	# i want full data to be an numpy array
	# my_mv_dat = ["movieId","title","genres"]
	# my_rating_dat = ["userId","movieId","rating"]
	# mv_m = mv_df[my_mv_dat].values
	# rt_m = rt_df[my_rating_dat].values
	# # get me all the movies that user 1-5 has rated
	# dat = []
	# count = 0
	# for mv_row in mv_m:
	# 	for user_row in rt_m:
	# 		if mv_row[0] == user_row[1]:
	# 			dat.append([count,mv_row[1],user_row[2],mv_row[2]])
	# 			count+=1
	# 			if count == 500:
	# 				pprint(dat)
	# 				return dat


	# pprint(dat)
	# return dat
	pass

def movie_recommender():
	#lets work with only ratings
	# so userID has rated a movie with some  movie ID


	pass





playing_around()











if __name__ == "__main__":
	print("in practice.py")
	def void():
		x = np.array([[2, 4, 3],[5,9,8]])
		y = np.array([[8, 4, 1], [2, 9, 1]])
		z = np.array([[5, 1, 2], [5, 19, 8],[8,4,5]])
		print(x.ndim)
		print(x)
		print(x.shape) # Shape is always tuple
		#print(x.reshape(3,2),"Altered Shape")

		#print(x.transpose()) # Transforming Rows Into Columns
		# Tensor has no dimensions eg [2],[5] , referred as a scalar

		print(5+x)
		print()
		print(5*x)
		print()
		print(5 - x)

		print(y + x)
		print()
		print(y * x)
		print()
		# print(y * z) # Not Possible (Because different dimensions)


		u = np.identity(6)
		print(u)

		z_inv = npla.inv(z)
		print(z * z_inv)



		print(np.sqrt(z))
