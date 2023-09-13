import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import json



df1 = pd.read_csv("bengaluru_house_prices.csv")
print(df1.head())
print(df1.shape)

print(df1['area_type'].value_counts())

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(df2.shape)

print(df2.isna().sum())

df3 = df2.dropna()
print(df3.isnull().sum())
print(df3.shape)

print("")
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.head(10))


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df3[~df3['total_sqft'].apply(is_float)].head(10)
print(df3.head(10))


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


df4 = df3.copy()
df4.total_sqft = df4["total_sqft"].apply(convert_sqft_to_num)
df4 = df4[df4["total_sqft"].notnull()]
df4.head(2)

# Feature Engineering
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
print(location_stats)

print("")
print(location_stats.values.sum())
print(len(location_stats[location_stats > 10]))

location_stats_less_than_10 = location_stats[location_stats <= 10]
print(location_stats_less_than_10)

df5["location"] = df5["location"].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

# Outlier Removal
print(df5[df5["total_sqft"] / df5["bhk"] < 300].head())
print(df5.shape)

df6 = df5[~(df5["total_sqft"] / df5["bhk"] < 300)]
print(df6.shape)

# Q1 = df6["price_per_sqft"].quantile(0.25)
# Q3 = df6["price_per_sqft"].quantile(0.75)
# IQR = Q3 - Q1
#
# lower_limit = Q1 - 1.5 * IQR
# upper_limit = Q3 + 1.5 * IQR
# df7 = df6[(df6["price_per_sqft"] > lower_limit) & (df6["price_per_sqft"] < upper_limit)]
# print(df7.head(10))
# print(df7.shape)


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_pps_outliers(df6)
print(df7.shape)


# Plotting the graph
# def plot_scatter_chart(df, location):
#     bhk2 = df[(df.location == location) & (df.bhk == 2)]
#     bhk3 = df[(df.location == location) & (df.bhk == 3)]
#     plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
#     plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
#     plt.xlabel("Total Square Feet Area")
#     plt.ylabel("Price (Lakh Indian Rupees)")
#     plt.title(location)
#     plt.legend()
#     plt.show()
#
#
# plot_scatter_chart(df7, "Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
print(df8.shape)

df9 = df8[df8.bath < df8.bhk+2]
print(df9.shape)

df10 = df9.drop(["size", "price_per_sqft"], axis=1)
print(df10)


# One hot encoding
dummies = pd.get_dummies(df10.location)
dummies.head(3)

df11 = pd.concat([df10, dummies.drop('other', axis=1)], axis=1)
print(df11.head())

df12 = df11.drop("location", axis=1)
print(df12.shape)

# Build a model
x = df12.drop("price", axis=1)
y = df12["price"]

# Train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Linear Regression
print("")
model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

# cross val score
cv = ShuffleSplit(random_state=0, n_splits=5, test_size=0.20)
print(cross_val_score(LinearRegression(), x, y, cv=cv))

# GridSearchCV
algos = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            # "normalize": [True, False]
        }
    },

    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [1, 2],
            "selection": ["random", "cyclic"]
        }
    },

    "DecisionTree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "criterion": ["mse", "friedman_mse"],
            "splitter": ["best", "random"]
        }
    }
}

scores = []
cv = ShuffleSplit(random_state=0, test_size=0.20, n_splits=5)
for i, v in algos.items():
    gs = GridSearchCV(v["model"], v["params"], cv=cv)
    gs.fit(x, y)
    scores.append({
        "Model": i,
        "BestScore": gs.best_score_,
        "BestParams": gs.best_params_
    })

df13 = pd.DataFrame(scores, columns=["Model", "BestScore", "BestParams"])
print(df13)
print(df13["BestScore"])


# Test the price for few properties
def price_predict(location, sqft, bhk, bath):
    loc_index = np.where(x.columns == location)[0][0]
    # print(loc_index)
    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1
    return model.predict([X])[0]


a = price_predict("1st Phase JP Nagar", 1000, 2, 2)
print(a)


with open("bangalore_home_prediction_model.pickle", "wb") as f:
    pickle.dump(model, f)


columns = {
    "data_columns": [col.lower() for col in x.columns]
}

with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

