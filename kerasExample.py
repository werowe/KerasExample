import keras
import pandas as pd



url = 'https://raw.githubusercontent.com/werowe/logisticRegressionBestModel/master/KidCreative.csv'

data = pd.read_csv(url, delimiter=',')

labels=data['Buy']
features = data.iloc[:,2:16]

model.fit(features, labels, epochs=10, batch_size=32)
one_hot_labels = keras.utils.to_categorical(labels, num_classes=2)




