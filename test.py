from sklearn.datasets import load_iris
from numpy import array
from random import shuffle

from neuralnet import NeuralNet, Layer, Node

data = load_iris()
target = data['target'].tolist()

actual = []

for i in target:
    actual.append([1 if i == j else 0 for j in xrange(0,3)])

dataset = zip(data['data'].tolist(), actual)
shuffle(dataset)
train = dataset[:101]
test = dataset[101:]

nn = NeuralNet()
nn.set_layers([
        Layer('input', 4),
        Layer('hidden', 10, "LReLU"),
        Layer('output', 3, "LReLU")
    ])


score = 0.0
for i in test:
    p = nn.predict(i)
    if p.index(max(p)) == i[1].index(1):
        score += 1
        
        
# Expect a value around 0.333, since there is a
# 1 in 3 chance to randomly guess correctly
print("Before: {}\n".format(score / len(test)))

nn.train(train, epochs = 1000, learning_rate=0.01)

score = 0.0
for i in test:
    p = nn.predict(i)
    if p.index(max(p)) == i[1].index(1):
        score += 1
        
# Expect a value much closer to 1.0, since, if
# all went well, the neural net now knows a thing
# or two about irises
print("After : {}\n".format(score / len(test)))

