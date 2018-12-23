# Kaggle_house_price_prediction
Solution proposal for the Kaggle competition for House Price Prediction

This proposal is based on 2 key ingredients:

Feature engineering
Model fitting

# Feature engineering
Feaure engineering is based on cleaning null data and removing columns where the amount of missing data is too high so that it is not worth to keep in the model.
There are a lot of categorical data with more than 2 possible values. Since getting all these features would get a too high feature space, we try to keep the dummy variables that are potentially valuable.
Thus instead of having for example (not real case) colour Blue Red Yellow Green, we might just need to keep information of Red / Not red. Afterwards the 10 independent featrues tha most correlate with the target variable are chosen to get into the model.

# Model fitting.
Finally, XGBoost was used for the prediction of the target variable SalePrice. Some other tecniques were tried providing worse results on the test set.
