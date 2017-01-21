import numpy as np
from scipy.stats.stats import pearsonr
from loaddata import output_data
import matplotlib.pyplot as plt


class Ccm:

    def __init__(self, time_series_data, target_data, dimension_E,
                 delta_T, lib_Sizes):
        self.time_series_data = time_series_data
        self.target_data = target_data
        self.dimension_E = dimension_E
        self.points_num_L = len(time_series_data)
        self.delta_T = delta_T
        self.lib_Sizes = lib_Sizes  # []

        self.start_point = delta_T * (dimension_E - 1) + 1
        self.end_point = self.points_num_L
        self.manifold_data = []
        self.manifold_data_num = self.points_num_L - \
            (dimension_E - 1) * delta_T
        self.correlations = []
        # record one situation in lib_Sizes
        self.selected_points_indices = []
        self.distances = []
        self.weights = []
        self.dist_indices = []  # the point itself is not include  in indices
        self.weight_indices = []
        self.estimate_results = []

    def empty_data(self):
        self.selected_points_indices = []
        self.distances = []
        self.weights = []
        self.dist_indices = []  # the point itself is not include  in indices
        self.weight_indices = []
        self.estimate_results = []

    def create_manifold(self):
        for t in xrange(self.start_point - 1, self.end_point):
            temp_manifold_data = []
            for i in xrange(self.dimension_E):
                temp_manifold_data.append(self.time_series_data[
                                          t - i * self.delta_T])
            self.manifold_data.append(temp_manifold_data)

    def select_manifold_data(self, lib_size):
        # default: choose points randomly
        indices = range(self.manifold_data_num)
        np.random.shuffle(indices)
        self.selected_points_indices = indices[:lib_size]

    def find_nearest_neighbor(self):
        selected_points = [self.manifold_data[indice]
                           for indice in self.selected_points_indices]
        array_selected_points = np.array(selected_points)
        array_manifold_data = np.array(self.manifold_data)
        temp_distances = []
        for index in range(self.manifold_data_num):
            # check whether the point itself in the selected points indice
            temp_selected_points_indices = []
            for i in self.selected_points_indices:
                if i != index:
                    temp_selected_points_indices.append(i)
                else:
                    continue
            temp_distances = [sum(
                (array_manifold_data[index] - array_selected_points[i])**2)
                ** 0.5 for i in range(len(temp_selected_points_indices))]
            self.distances.append(temp_distances)
            self.dist_indices.append(temp_selected_points_indices)

    def create_weights(self):
        for noi, dist in enumerate(self.distances):
            temp_index = sorted(range(len(dist)), key=lambda k: dist[k])
            temp_index = temp_index[:self.dimension_E + 1]
            temp_weight_indices = [self.dist_indices[noi][i]
                                   for i in temp_index]
            self.weight_indices.append(temp_weight_indices)
            temp_distances = [dist[d] for d in temp_index]
            d1 = temp_distances[0]
            # check weights
            if float(d1) == 0.0:
                uis = []
                for i in range(len(temp_index)):
                    if float(temp_distances[i]) == 0.0:
                        uis.append(np.exp(-1))
                    else:
                        uis.append(0.0)
                N = sum(uis)
                temp_weights = [ui / float(N) for ui in uis]
                self.weights.append(temp_weights)
            else:
                uis = [np.exp(-di / d1) for di in temp_distances]
                N = sum(uis)
                temp_weights = [ui / float(N) for ui in uis]
                self.weights.append(temp_weights)

    def estimate_y(self):
        for temp_indices, temp_weights in zip(self.weight_indices,
                                              self.weights):
            pred_y = 0
            for ii, wi in zip(temp_indices, temp_weights):
                pred_y += self.target_data[ii] * wi
            self.estimate_results.append(pred_y)

    def compute_correlations(self):
        self.create_manifold()
        true_y = self.target_data[self.start_point - 1: self.end_point]
        array_true_y = np.array(true_y)
        for lib_size in self.lib_Sizes:
            self.select_manifold_data(lib_size)
            self.find_nearest_neighbor()
            self.create_weights()
            self.estimate_y()
            # pearsonr
            p_value, _ = pearsonr(array_true_y,
                                  np.array(self.estimate_results))
            self.correlations.append(p_value)
            self.empty_data()


def main(start_size, end_size, interval):
    # read data
    kyoto_nov, osaka_nov, kobe_nov = output_data()
    # tokyo_nov = data[0][0]
    kyoto_nov_temp = []
    osaka_nov_temp = []
    # tokyo_nov_temp = []
    for i in kyoto_nov:
        kyoto_nov_temp.append(float(i[5]))
    for i in osaka_nov:
        osaka_nov_temp.append(float(i[5]))

    lib_Sizes = range(start_size, end_size, interval)
    temp_ccm = Ccm(kyoto_nov_temp, osaka_nov_temp, dimension_E=2,
                   delta_T=1, lib_Sizes=lib_Sizes)
    temp_ccm.compute_correlations()
    f, ax = plt.subplots()
    ax.set_title('Osaka->Kyoto')
    ax.set_xlabel('Library Size')
    ax.set_ylabel('Coefficient')
    line, = ax.plot(lib_Sizes, temp_ccm.correlations, 'r', lw=1,
                    label='Osaka->Kyoto')
    plt.show()


if __name__ == '__main__':
    main(60, 100, 1)
