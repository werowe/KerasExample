import keras
import pandas as pd

url = 'https://raw.githubusercontent.com/werowe/logisticRegressionBestModel/master/KidCreative.csv'

data = pd.read_csv(url, delimiter=',')

y=data['Buy']
x = data.iloc[:,2:16]

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=seed)

model = Sequential()


