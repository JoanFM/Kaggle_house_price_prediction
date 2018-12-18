#This script contains two parts which should be split in two modules, one for feature engineering and one for actual
#model training and evaluation

###################################################################################################################
###################################################################################################################
###################################FEATURE ENGINEERING ##### ######################################################
###################################################################################################################
###################################################################################################################
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from   sklearn import preprocessing
from   typing import Any
import numpy as np
from   scipy import stats

#number of numerical values to consider based on the correlation with the target
NUMBER_OF_INDEPENDENT_VARIABLES_TO_CONSIDER = 10

#for numerical data replace by the mean, for categorical, replace with None
def replace_nulls(column) :
    if column.dtype == "object" :
        if column.name in ["SaleType", "Exterior2nd", "Exterior1st"] :
            null_column = column.fillna("Other")
        elif column.name in ["Utilities", "KitchenQual", "Functional", "Electrical"] :
            null_column = column.fillna(column.mode()[0])
        elif column.name in ["BsmtExposure"] :
            null_column = column.fillna("No")
        elif column.name in ["BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond"] :
            null_column = column.fillna("NA")
        else :
            null_column = column.fillna("None")
    else :
        #if it is a numerical column you fill the null values with the mean.
        null_column = column.fillna(column.mean())

    return null_column

PVALUE_THRESHOLD = 0.05
#get dummies that bring value to the model
def get_valuable_dummies(column_train, column_test, df_train) :
    dummies_train = pd.get_dummies(column_train, prefix = column_train.name)
    dummies_test  = pd.get_dummies(column_test, prefix = column_test.name)
    for k in column_train.unique() :
        tag = column_train.name + "_" + k
        if tag in dummies_test.columns :
            cat1 = df_train[column_train == k]["SalePrice"]
            cat2 = df_train[column_train != k]["SalePrice"]
            #split in two sets depending on the value of the column and k
            ttest_result = stats.ttest_ind(cat1, cat2, equal_var = False)
            #if the pvalue is above the threshold, discard that category as 
            #we discard that the mean is not affected by it
            if ttest_result.pvalue > PVALUE_THRESHOLD :
                dummies_train = dummies_train.drop(tag, axis = 'columns')
                dummies_test = dummies_test.drop(tag, axis = 'columns')
        else :
            dummies_train = dummies_train.drop(tag, axis = 'columns')

    #clean to have model param = to test param
    for k in column_test.unique() :
        tag = column_test.name + "_" + k
        if (tag in dummies_test.columns) and (tag not in dummies_train.columns) :
            dummies_test = dummies_test.drop(tag, axis = 'columns')

    return dummies_train, dummies_test

#read data
train = pd.read_csv('../input/train.csv', index_col = 0)  # type: Any
test  = pd.read_csv('../input/test.csv', index_col = 0)

#remove outliers
train = train[train.GrLivArea < 4000]

tmp_saleprice = train["SalePrice"]

#get total as a concatenation of train and test to consider null values on both sets
total = pd.concat([train.drop("SalePrice", axis = 1), test])

MIN_PERCENTAGE_NO_NULL_RECORDS_PER_COLUMN = 0.95
threshold = MIN_PERCENTAGE_NO_NULL_RECORDS_PER_COLUMN * total.shape[0]

#just keep columns with more than 95% of valid data
total = total.dropna(thresh = threshold, axis = 'columns')
total = total.apply(lambda col: replace_nulls(col), 0)

#split again the train and test set, notice that the drop of null values was made on the columns
train = total[0:train.shape[0]]
train = pd.concat([train, tmp_saleprice], axis = 1)
test  = total[train.shape[0]:(train.shape[0] + test.shape[0])]

#log1p transformation on SalePrice to normalize and get proper value because
#error will be based on the log
saleprice    = train["SalePrice"]
no_null      = train.drop("SalePrice", axis = 1)
saleprice    = np.log1p(saleprice)
no_null      = pd.concat([no_null, saleprice], axis = 1)
no_null_test = test

#get only numerical features that have high correlation with SalePrice
numerical_features      = no_null.select_dtypes(exclude = ["object"]).columns
numerical_features_test = no_null_test.select_dtypes(exclude = ["object"]).columns
categorical_features    = no_null.select_dtypes(include = ["object"]).columns
#detect the number of numerical features to use in the model depending on the correlations
k = NUMBER_OF_INDEPENDENT_VARIABLES_TO_CONSIDER
corrmat = no_null[numerical_features].corr()
numerical_cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
numerical_cols = numerical_cols.drop("SalePrice")

#get the sets with no null values and with the highly correlated numerical features
no_null = pd.concat([no_null[categorical_features], no_null[numerical_cols]], axis = 1)
no_null = pd.concat([no_null, saleprice], axis = 1)
no_null_test = pd.concat([no_null_test[categorical_features], no_null_test[numerical_cols]], axis = 1)

#for training data I want just the dummies that are interested
cat_columns = no_null[categorical_features]
clean       = no_null[numerical_cols]
clean_test  = no_null_test[numerical_cols]

for cat in cat_columns:
    valuable_dummies_train, valuable_dummies_test = get_valuable_dummies(no_null[cat],no_null_test[cat], no_null)
    clean = pd.concat([clean, valuable_dummies_train], axis = 1)
    clean_test = pd.concat([clean_test, valuable_dummies_test], axis = 1)

#add the log1p(saleprice)
clean  = pd.concat([clean, saleprice], axis = 1)

#now standarize numerical features
standarizer = preprocessing.StandardScaler()
clean.loc[:, numerical_cols]      = standarizer.fit_transform(clean.loc[:, numerical_cols])
clean_test.loc[:, numerical_cols] = standarizer.fit_transform(clean_test.loc[:, numerical_cols])

clean.to_csv('clean_train.csv')
clean_test.to_csv('clean_test.csv')

###################################################################################################################
###################################################################################################################
###################################MODEL TRAINING AND FITING ######################################################
###################################################################################################################
###################################################################################################################

from   math import sqrt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from   sklearn.metrics import make_scorer, mean_squared_error
from   typing import Any
import xgboost as xgb

#obtain the clean datasets
train_dataframe = pd.read_csv('clean_train.csv', index_col = 0)  # type: Any
test_dataframe  = pd.read_csv('clean_test.csv', index_col = 0)  # type: Any

#get X_train , Y_train and X_test
Y_train = train_dataframe["SalePrice"]
X_train = train_dataframe.drop("SalePrice", axis = 1)
X_test  = test_dataframe

#build and fit model
model_xgb = xgb.XGBRegressor(learning_rate = 0.05, n_estimators = 300,
                             reg_alpha = 0.5, reg_lambda = 0.5,
                             subsample = 0.5, silent = 1, nthread = -1)

model_xgb.fit(X_train, Y_train)

y_train_pred = model_xgb.predict(X_train)
y_test_pred  = model_xgb.predict(X_test)

#make predictions
prediction = pd.DataFrame(np.expm1(y_test_pred), columns = ['SalePrice'])
prediction = prediction.set_index(test_dataframe.index)

prediction.to_csv('prediction.csv')