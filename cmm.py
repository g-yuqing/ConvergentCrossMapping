import numpy as np
from scipy.stats.stats import pearsonr
from loaddata import output_data
import matplotlib.pyplot as plt


class Ccm:

    def __init__(self, time_series_data, target_data, dimension_E,
                 points_num_L, delta_T):
        self.time_series_data = time_series_data  # []
        self.target_data = target_data
        self.dimension_E = dimension_E
        self.points_num_L = points_num_L
        self.delta_T = delta_T

        self.start_point = delta_T * (dimension_E - 1) + 1
        self.end_point = points_num_L
        self.manifold_data = []
        self.manifold_data_num = points_num_L - (dimension_E - 1) * delta_T
        self.points_distances = [
            [0 for x in xrange(self.manifold_data_num)]
            for y in xrange(self.manifold_data_num)]
        self.weights = []
        self.indices = []
        self.estimate_results = []
        self.correlation = 0

    def create_manifold(self):
        for t in xrange(self.start_point - 1, self.end_point):
            temp_manifold_data = []
            for i in xrange(self.dimension_E):
                temp_manifold_data.append(
                    self.time_series_data[t - i * self.delta_T])
            self.manifold_data.append(temp_manifold_data)

    def find_nearest_neighbor(self):
        array_manifold_data = np.array(self.manifold_data)
        for i in xrange(self.manifold_data_num):
            for j in xrange(i, self.manifold_data_num):
                # calculate distances
                temp_distance = sum((array_manifold_data[i] -
                                     array_manifold_data[j])**2.0)**0.5
                self.points_distances[i][j] = temp_distance
                self.points_distances[j][i] = temp_distance

    def create_weights(self):
        for distances in self.points_distances:
            # calculate indices
            temp_index = sorted(range(len(distances)),
                                key=lambda k: distances[k])
            temp_index = temp_index[1:self.dimension_E + 2]  # E+1 are be used
            self.indices.append(temp_index)
            # calculate weights
            temp_distances = [distances[d] for d in temp_index]  # E+1 nearest
            d1 = temp_distances[0]
            uis = [np.exp(-di / d1) for di in temp_distances]
            N = sum(uis)
            temp_weights = [ui / float(N) for ui in uis]
            self.weights.append(temp_weights)

    def estimate_y(self):
        for temp_indices, temp_weights in zip(self.indices, self.weights):
            pred_y = 0
            for ii, wi in zip(temp_indices, temp_weights):
                pred_y += self.target_data[ii] * wi
            # in estimate_results, start_point-1 maps to 0
            self.estimate_results.append(pred_y)

    def compute_correlation(self):
        true_y = self.target_data[self.start_point - 1: self.end_point]
        pred_y = self.estimate_results
        # pearsonr
        _, p_value = pearsonr(np.array(true_y), np.array(pred_y))
        self.correlation = p_value


def main(start_size, end_size, interval):
    correlations = []
    # read data
    data = output_data()
    kyoto_nov = data[1][0]
    osaka_nov = data[1][1]
    kyoto_nov_temp = []
    osaka_nov_temp = []
    for i in kyoto_nov:
        kyoto_nov_temp.append(float(i[5]))
    for i in osaka_nov:
        osaka_nov_temp.append(float(i[5]))

    for i in xrange(start_size, end_size, interval):
        temp_kyoto = kyoto_nov_temp[:i]
        temp_osaka = osaka_nov_temp[:i]
        temp_len = len(temp_kyoto)
        temp_ccm = Ccm(temp_kyoto, temp_osaka, dimension_E=2,
                       points_num_L=temp_len, delta_T=1)
        temp_ccm.create_manifold()
        temp_ccm.find_nearest_neighbor()
        temp_ccm.create_weights()
        temp_ccm.estimate_y()
        temp_ccm.compute_correlation()
        correlations.append(temp_ccm.correlation)
    xs = range(start_size, end_size, interval)
    f, ax = plt.subplots()
    ax.set_title('Kyoto_Osaka')
    ax.set_xlabel('Library Size')
    ax.set_ylabel('Coefficient')
    line, = ax.plot(xs, correlations, 'r', lw=1, label='Kyoto->Osaka')
    plt.show()


if __name__ == '__main__':
    main(50, 56, 1)
