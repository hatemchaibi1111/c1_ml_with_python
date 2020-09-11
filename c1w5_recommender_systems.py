import os
import urllib.request
from zipfile import ZipFile
import pandas as pd

filenames = ['links.csv', 'movies.csv', 'ratings.csv', 'tags.csv']
if not all([os.path.exists('moviedataset/ml-latest/' + x) for x in filenames]):
    filename = 'moviedataset.zip'
    if not os.path.exists(filename):
        print('Downloading moviedataset.zip..')
        url = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/'
               'moviedataset.zip')
        urllib.request.urlretrieve(url, filename)
        print('Done.')
    print('Unziping moviedataset.zip..')

    with ZipFile('moviedataset.zip', 'r') as zipObj:
        listOfFileNames = zipObj.namelist()
        #  Extract all the contents of zip file in current directory
        for file in listOfFileNames:
            zipObj.extract(file, path='moviedataset/')
    print('Done.')


# Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('moviedataset/ml-latest/movies.csv')
# Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('moviedataset/ml-latest/ratings.csv')

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df['genro calles'] = movies_df.genres.str.split('|')

moviesWithGenres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres'].split('|'):
        moviesWithGenres_df.at[index, genre] = 1
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
ratings_df = ratings_df.drop('timestamp', 1)

# Content-Based recommendation system

userInput = [
            {'title': 'Breakfast Club, The', 'rating': 5},
            {'title': 'Toy Story', 'rating': 3.5},
            {'title': 'Jumanji', 'rating': 2},
            {'title': "Pulp Fiction", 'rating': 5},
            {'title': 'Akira', 'rating': 4.5}
         ]
inputMovies = pd.DataFrame(userInput)

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies = userMovies.reset_index(drop=True)
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genro calles', 1).drop('year', 1).drop('genres', 1)

# Dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1).drop('genro calles', 1)

# Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

print(recommendationTable_df.head())
pd.set_option('display.max_columns', 500)
# The final recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])

# COLLABORATIVE FILTERING
print('COLLABORATIVE FILTERING')

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movies_df = pd.read_csv('moviedataset/ml-latest/movies.csv')
ratings_df = pd.read_csv('moviedataset/ml-latest/ratings.csv')
inputMovies = pd.DataFrame(userInput)


inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year', 1)

# Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
# Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
# Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)

# Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# For every user group in our subset
for name, group in userSubsetGroup:
    # Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(
        nRatings)

    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0




