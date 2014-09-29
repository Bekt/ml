import time
import os

from functions import Objective
from supervised import Supervised


def _write_iter(name, col):
    with open(name, 'w') as f:
        f.write(os.linesep.join(str(v) for v in col))

class NeuralNet(Supervised):

    def __init__(self, optimizer=None, iterations=1000):
        self._layers = []
        self._optimizer = optimizer
        self._iterations = iterations
        self._weight_count = None

    def train(self, features, labels):
        assert(features.data.shape[0] == labels.data.shape[0])
        portion = int(features.rows() * 0.75)
        train_fe = features.sub_rows(0, portion)
        train_la = labels.sub_rows(0, portion)
        test_fe = features.sub_rows(portion, features.rows())
        test_la = labels.sub_rows(portion, labels.rows())

        obj = Objective(train_fe, train_la, self)
        # Inject the objective into the optimizer here
        # because there is a 3-way cycle:
        # Objective -> Learner -> Optimizer -> Objective
        self._optimizer._obj = obj
        self.write_times(train_fe, train_la, test_fe, test_la)

    def write_times(self, train_fe, train_la, test_fe, test_la):
        plots_train_t = []
        plots_train_e = []
        plots_test_t = []
        plots_test_e = []
        start = int(time.time())
        for i in range(self._iterations):
            err = self._optimizer.iterate()
            # HC is ridic slow.
            mod = 1 if self._optimizer.name == 'hillclimber' else 25
            if i % mod == 0:
                t = int(time.time()) - start
                if t >= 1800:
                    # 30 minutes?
                    break
                plots_train_t.append(t)
                plots_train_e.append(err)
                sse = self.measure_sse(test_fe, test_la)
                plots_test_t.append(t)
                plots_test_e.append(sse)
                print(i, t, err, sse)
        _write_iter('train_times_' + self._optimizer.name + '.txt', plots_train_t)
        _write_iter('train_errors_' + self._optimizer.name + '.txt', plots_train_e)
        _write_iter('test_times_' + self._optimizer.name + '.txt', plots_test_t)
        _write_iter('test_errors_' + self._optimizer.name + '.txt', plots_test_e)

    def predict(self, inp):
        if not self._layers:
            raise ValueError('Layers is empty.')
        act = inp
        for layer in self._layers:
            layer.feed(act)
            act = layer._activation
        return act

    def set_params(self, vector):
        assert(vector.size == self.weights_count())
        prev = 0
        for layer in self._layers:
            w, b = layer._weights, layer._bias
            layer._weights = vector[prev:prev+w.size].reshape(w.shape)
            prev += w.size - 1
            layer._bias = vector[prev:prev+b.size].reshape(b.shape)
            prev += b.size - 1

    def weights_count(self):
        if not self._weight_count:
            self._weight_count = sum([la._weights.size + la._bias.size for la in self._layers])
        return self._weight_count
