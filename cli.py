import sys
import pandas as pd

# print(sys.argv)

df = pd.read_csv(sys.argv[1])
print(df.head())



# df1 = pd.read_csv('jigsaw-toxic-comment-classification-challenge/train.csv/train.csv')
# print(df1.head())