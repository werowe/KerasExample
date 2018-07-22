from keras.models import Sequential
import pandas as pd

url = 'https://raw.githubusercontent.com/werowe/logisticRegressionBestModel/master/KidCreative.csv'

data = pd.read_csv(url, delimiter=',')

labels=data['Buy']
features = data.iloc[:,2:16]

model = Sequential()
model.compile(optimizer='rmsprop' ,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=32)
model.evaluate(labels, features, batch_size=128)

model.summary()



