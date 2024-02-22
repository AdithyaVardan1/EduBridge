import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('path_to_dataset.csv')

df = df.dropna()

df = df[['Question', 'Answer']]

train_df, eval_df = train_test_split(df, test_size=0.1)
