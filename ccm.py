import numpy as np
from scipy.stats.stats import pearsonr
from loaddata import output_data
import matplotlib.pyplot as plt


class Ccm:

    def __init__(self, time_series_data, target_data, dimension_E,
                 points_num_L, delta_T, lib_Size=0):
        self.time_series_data = time_series_data  # []
        self.target_data = target_data
        self.dimension_E = dimension_E
        self.points_num_L = points_num_L
        self.delta_T = delta_T
        self.lib_Size = lib_Size

        self.start_point = delta_T * (dimension_E - 1) + 1
        self.end_point = points_num_L
        self.manifold_data = []
        self.manifold_data_num = points_num_L - (dimension_E - 1) * delta_T
        self.selected_manifold_data = []
        self.points_distances = [
            [0 for x in xrange(self.manifold_data_num)]
            for y in xrange(self.manifold_data_num)]
        self.selected_points_distances = []
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

    def select_mnifold(self):
        # choose later points with the number of lib_size
        last_point_indice = self.manifold_data_num - self.lib_Size
        selected_points_num = last_point_indice + 1
        for i in xrange(selected_points_num):
            self.selected_manifold_data.append(
                self.manifold_data[i:i + self.lib_Size])

    def find_nearest_neighbor_selected(self):
        for selected_units in self.selected_manifold_data:
            array_selected_units = np.array(selected_units)
            temp_distance = [sum((array_selected_units[0] -
                                  array_selected_units[i])**2)**0.5
                             for i in range(1, self.lib_Size)]
            self.selected_points_distances.append(temp_distance)

    def find_nearest_neighbor_all(self):
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
            for i in range(len(temp_index)):
                if 0.0 == float(distances[temp_index[i]]):
                    continue
                else:
                    temp_index = temp_index[i:self.dimension_E + 1 + i]
                    break
                # temp_index = temp_index[1:self.dimension_E + 2]  # E+1 are be
                # 'used
            self.indices.append(temp_index)
            # calculate weights
            temp_distances = [distances[d] for d in temp_index]  # E+1 nearest
            d1 = temp_distances[0]
            uis = [np.exp(-di / d1) for di in temp_distances]
            N = sum(uis)
            temp_weights = [ui / float(N) for ui in uis]
            self.weights.append(temp_weights)

    def create_weights2(self):
        for distances in self.points_distances:
            # calculate indices
            temp_index = sorted(range(len(distances)),
                                key=lambda k: distances[k])
            temp_index = temp_index[1: self.dimension_E + 2]
            self.indices.append(temp_index)
            # calculate weights
            temp_distances = [distances[d] for d in temp_index]
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
        p_value, _ = pearsonr(np.array(true_y), np.array(pred_y))
        self.correlation = p_value


def main_all_points(start_size, end_size, interval):
    correlations = []
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

    for i in xrange(start_size, end_size, interval):
        temp_kyoto = kyoto_nov_temp[:i]
        temp_osaka = osaka_nov_temp[:i]
        # temp_tokyo = tokyo_nov_temp[:i]
        temp_len = len(temp_kyoto)
        temp_ccm = Ccm(temp_kyoto, temp_osaka, dimension_E=2,
                       points_num_L=temp_len, delta_T=1)
        # temp_ccm = Ccm(temp_kyoto, temp_tokyo, dimension_E=2,
        #                points_num_L=temp_len, delta_T=1)
        temp_ccm.create_manifold()
        temp_ccm.find_nearest_neighbor_all()
        temp_ccm.create_weights2()
        temp_ccm.estimate_y()
        temp_ccm.compute_correlation()
        correlations.append(temp_ccm.correlation)
    xs = range(start_size, end_size, interval)
    f, ax = plt.subplots()
    ax.set_title('Osaka->Kyoto')
    ax.set_xlabel('Library Size')
    ax.set_ylabel('Coefficient')
    line, = ax.plot(xs, correlations, 'r', lw=1, label='Osaka->Kyoto')
    plt.show()


def main_selected_points(start_size, end_size, interval):
    correlations = []
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

    for i in xrange(start_size, end_size, interval):
        # temp_tokyo = tokyo_nov_temp[:i]
        temp_len = len(kyoto_nov_temp)
        temp_ccm = Ccm(kyoto_nov_temp, osaka_nov_temp, dimension_E=2,
                       points_num_L=temp_len, delta_T=1, lib_Size=i)
        # temp_ccm = Ccm(temp_kyoto, temp_tokyo, dimension_E=2,
        #                points_num_L=temp_len, delta_T=1)
        temp_ccm.create_manifold()
        temp_ccm.select_mnifold()
        temp_ccm.find_nearest_neighbor_selected()
        temp_ccm.create_weights2()
        temp_ccm.estimate_y()
        temp_ccm.compute_correlation()
        correlations.append(temp_ccm.correlation)
    xs = range(start_size, end_size, interval)
    f, ax = plt.subplots()
    ax.set_title('Osaka->Kyoto')
    ax.set_xlabel('Library Size')
    ax.set_ylabel('Coefficient')
    line, = ax.plot(xs, correlations, 'r', lw=1, label='Osaka->Kyoto')
    plt.show()


if __name__ == '__main__':
    main_all_points(25, 60, 1)
    # main_selected_points(25, 60, 1)
