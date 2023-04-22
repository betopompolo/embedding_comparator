from typing import List
import pandas as pd

# dataframe = pd.read_csv('datasets/queries.csv')
# filtered_queries = dataframe[dataframe['Language'].isin(['Python', 'Java'])]
# filtered_queries.to_csv('datasets/filtered_queries.csv', index=False, header=False)

def get_validation_queries_df():
  dataframe = pd.read_csv('datasets/queries.csv')
  return dataframe[dataframe['Language'].isin(['Python', 'Java'])]

df = get_validation_queries_df()
for index, row in df.iterrows():
  print(row['Language']) # TODO: Yield mapped data to the consumer
