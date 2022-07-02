import pandas as pd

df = pd.read_json("./yelp_academic_dataset_user.json", lines=True)

df.dropna()
print(df.head())
rest = df.sample(200)
rest.to_csv('./user.csv')