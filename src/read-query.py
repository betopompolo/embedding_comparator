import pandas as pd

dataframe = pd.read_csv('datasets/queries.csv')
filtered_queries = dataframe[dataframe['Language'].isin(['Python', 'Java'])]
filtered_queries.to_csv('datasets/filtered_queries.csv', index=False, header=False)

