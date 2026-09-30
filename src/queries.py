from connection_db import mycol
import pymongo
import pandas as pd


group_lang_movie = mycol.aggregate([{
  "$group": {
     "_id" : "$original_language",
     "count": {"$sum": 1},
     
  }},
  {
    "$sort" : {"count" : -1}
  }
  ])
df1 = pd.DataFrame(list(group_lang_movie))
df1.columns = ["language", "count"]   


high_rated_movies = mycol.find({'vote_average' : 8, 'vote_count' : {"$gt" : 1000}})


avg_budget = mycol.aggregate([{
  "$group" : {"_id" : "$release_year",
      "averageBudget" : {"$avg" : "$budget"},
      "count" : {"$sum": 1}}
  },
  {
    "$sort": {"count": -1}
  }])

df2 = pd.DataFrame(high_rated_movies)

df3 = pd.DataFrame(avg_budget)
df3.columns = ['release_year', 'budget', 'count']
print(df1)
print(df2)
print(df3)