import numpy as np


class RBFN:

    def __init__(self, input_x, output_y, dimesion_E, center_Num):
        self.input_x = input_x
        self.output_y = output_y
        self.dimension_E = dimesion_E
        self.center_Num = center_Num
        self.ro = 0
        self.centers = [np.random.uniform(-1, 1, dimesion_E)
                        for i in xrange(center_Num)]
        self.weights = np.random.random((center_Num, dimesion_E))
        self.neurons = np.zeros((input_x.shape[0], center_Num), float)

    def rbf_func(self, c, d):
        beta = -1 / float(2 * self.ro**2)
        return np.exp(beta * np.linalg.norm(c - d)**2)

    def calculate_centers(self):
        pass

    def calculate_ro(self):
        pass

    def calculate_neurons(self, input_x):
        for c_index, c_val in enumerate(self.centers):
            for x_index, x_val in enumerate(input_x):
                self.neurons[x_index, c_index] = self.rbf_func(c_val, x_val)

    def calculate_weights(self, input_x, output_y):
        self.weights = np.dot(np.linalg.pinv(self.neurons), output_y)

    def predict_results(self, input_x):
        pred_y = np.dot(self.neurons, self.weights)
        return pred_y


class Cms:

    def __init__(self, time_series_data, target_data, dimension_E,
                 delta_T):
        self.time_series_data = time_series_data
        self.target_data = target_data
        self.dimension_E = dimension_E
        self.points_num_L = len(time_series_data)
        self.delta_T = delta_T

        self.start_point = delta_T * (dimension_E - 1) + 1
        self.end_point = self.points_num_L
        self.manifold_data_x = []
        self.manifold_data_y = []

    def create_manifold(self):
        for t in xrange(self.start_point - 1, self.end_point):
            temp_manifold_data_x = []
            temp_manifold_data_y = []
            for i in xrange(self.dimension_E):
                temp_index = t - i * self.delta_T
                temp_manifold_data_x.append(self.time_series_data[temp_index])
                temp_manifold_data_y.append(self.target_data[temp_index])
            self.manifold_data_x.append(temp_manifold_data_x)
            self.manifold_data_y.append(temp_manifold_data_y)

    def calculate:
        for i in xrange(self.points_num_L):
            temp_data = np.zeros((self.points_num_L - 1, self.dimension_E))
            for j in xrange(self.points_num_L):
                if i == j:
                    continue
                else:
                    temp_data[j] = self.manifold_data_x[j]

            RBFN()
