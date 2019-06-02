import tensorflow as tf
import numpy as np
from mcts.record.Recorder import Recorder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import backend
 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# 실행할 때마다 같은 결과를 출력하기 위한 seed 값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

recorder = Recorder()
json = recorder.load('data/190601054701.json')
x = [[float(obj['param'][0]), float(obj['param'][1]), float(obj['param'][2])] for obj in json]
y = [float(obj['value']) for obj in json]
# x,y의 데이터 값
print(x, len(x))
print(y, len(y))



x_data = np.array(x)
y_data = np.array(y)#.reshape(len(y), 1)


model = Sequential()
model.add(Dense(3, input_shape=(3,), activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(1) )

# sgd = SGD(lr=0.1)
model.compile(loss='mse', optimizer=Adam(lr=0.01),  metrics=[rmse])

model.fit(x_data, y_data, batch_size=64, nb_epoch=50000)
final = []
testValues = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for value1 in range(0,11):
    for value2 in range(0,11):
        for value3 in range(0,11):
            a = testValues[value1]
            b = testValues[value2]
            c = testValues[value3]
            v = model.predict_proba(np.array([[a,b,c]]))
            final.append({"value":v, "param":[a,b,c]})
print(final)
print([val['value'][0][0] for val in final])
print(max(final, key=lambda v: v['value']))