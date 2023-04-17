from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import random

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})

enc = OneHotEncoder()
enc.fit(data[['whoAmI']])

one_hot = enc.transform(data[['whoAmI']])
cols = enc.get_feature_names_out(['whoAmI'])

one_hot_df = pd.DataFrame(one_hot.toarray(), columns=cols)
print(one_hot_df.head())