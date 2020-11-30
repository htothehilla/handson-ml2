import os
import tarfile
import urllib


import os

import os
os.chdir('/Users/xxx/PycharmProjects/handson-ml2/datasets')
cwd = os.getcwd()

#downloading dataset
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(cwd, "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
#provides top five rows

housing.info()
#provides information of data

housing.describe()
#summary stats e.g mean and stuff

#histogram - bins (int or sequence or str, optional)
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

#create a test set
import numpy as np

#creates test set
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)

#To have a stable train/test split even after updating the dataset,
#a common solution is to use each instance’s identifier to decide
#whether or not it should go in the test set
#(assuming instances have a unique and immutable identifier).

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#test set doesn't have id, so they uses houses longitudinal and latitude

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

#spliting dataset
# features -. set random generator seed + mulitiple datasets with an identical number of rows
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#checking if representative (create hist, creating continous attribute to a income catgeory
# five labels for each variable
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()

#stratified sampling based - reflect population
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#shows you that it has SS worked by looking like a variable

strat_test_set["income_cat"].value_counts() / len(strat_test_set)


#remove the income_cat attribute so the data is back to its original state:

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#put training set to aside
housing = strat_train_set.copy()


##Discover (if data is big you can do a exploration set)

#visulaising training set, alpha makes it more visible
#s = population and c = colour of median house prices
housing.plot(kind="scatter",x="longitude",y="latitude", alpha=0.1,
    s=housing["population"]/100,label="population",figsize=(10,7),
    c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,
)
plt.legend()

#looking for correlations
#only measures linear relationships
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#map correlations  by plotting every numerical attribute


from pandas.plotting import scatter_matrix
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

#zoom in on median income
#correlation strong (upward trend but plot reveals certain quirks and you don't want algorthim to repeat it

housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)

#checking out new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#correlation matrix

corr_matrix =housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#preparing data _> make functions

#revet to clean training set -> making a copy of training set _>
# removing lavels as you don;t want to apply same transformations to the predictors
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#data cleaning
#choose option 1
housing.dropna(subset=["total_bedrooms"])

#missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

#median can only be computed with numbers need to create a
#copy of the data without text attribute for ocean_proximity

housing_num = housing.drop("ocean_proximity",axis=1)

#
imputer.fit(housing_num)
#the imputer  computed a median of each attribute
# and stored the resukt in its stats intance variable
# safe to also add imputer on other variables

imputer.statistics_
housing_num.median().values

#replacing missing values with learned medians
x=imputer.transform(housing_num)

housing_tr = pd.DataFrame(x, columns=housing_num.columns,index=housing_num.index)


#text attributes
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

#Most Machine Learning algorithms prefer to work with numbers, so let’s
#convert these categories from text to numbers. For this,
#we can use Scikit-Learn’s OrdinalEncoder class:19

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

#get the names of categories
ordinal_encoder.categories_

#The new attributes
#are sometimes called dummy attributes.Scikit - Learn provides a OneHotEncoder
#class to convert categorical values into one-hot vectors:20

#we need dummy variables as the distcintion between 1, 2 are not the same conceptually for island, near bay...

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

#2d array to host the data
housing_cat_1hot.toarray()

#can find catergory names

cat_encoder.categories_

#custome transformers
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#transforming pipelines making the values to scale

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)