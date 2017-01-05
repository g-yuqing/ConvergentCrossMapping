import csv
import numpy as np
import matplotlib.pyplot as plt


def loadcsv(filename):
    with open(filename, 'rb') as f:
        lines = csv.reader(f)
        rows = []
        for row in lines:
            rows.append(row)
    return rows


def reshape_data(data):
    kyoto = []
    osaka = []
    kobe = []
    # data start from 7th line
    data = data[6:]
    for data_unit in data:
        kyoto_temp = []
        osaka_temp = []
        kobe_temp = []
        for i in range(len(data_unit)):
            if i % 18 == 1:  # cloud 1,19,37
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 4:  # rain 4,22,40
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 8:  # air pressure 8,26,44
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 11:  # wind 11,29,47
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 13:  # wind direction 13,31,49
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 16:  # temperature 16,34,52
                if (i - 1) / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if (i - 1) / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if (i - 1) / 18 == 2:
                    kobe_temp.append(data_unit[i])
        kyoto.append(kyoto_temp)
        osaka.append(osaka_temp)
        kobe.append(kobe_temp)
    return kyoto, osaka, kobe  # cloud,rain,pressure,wind,wd,temperature


def output_data():
    datafiles = ['data11e.csv', 'data11w.csv']
    # data will be stored in the form as 11e|11w
    # -> three areas in east/west -> each day's data -> 6 kinds
    data = []
    for filename in datafiles:
        temp_data = loadcsv(filename)
        region1, region2, region3 = reshape_data(temp_data)
        data.append([region1, region2, region3])  # data shape[2,3,600+,6]
    return data


def correlate_test():
    data = loadcsv()
    kyoto, osaka, kobe = reshape_data(data)
    rain_kyoto = []
    temp_kyoto = []
    for i in kyoto:
        rain_kyoto.append(float(i[1]))
        temp_kyoto.append(float(i[5]))
    rain_osaka = []
    temp_osaka = []
    for i in osaka:
        rain_osaka.append(float(i[1]))
        temp_osaka.append(float(i[5]))
    rain_kobe = []
    temp_kobe = []
    for i in kobe:
        rain_kobe.append(float(i[1]))
        temp_kobe.append(float(i[5]))

    co = np.correlate(temp_kyoto, temp_kobe, 'full')
    print(np.argmax(co))
    # cloud,rain,pressure,wind,wd,temperature
    plt.subplot(3, 1, 1)
    plt.scatter(temp_kyoto[1:], temp_kyoto[:-1])
    plt.subplot(3, 1, 2)
    plt.scatter(temp_kobe[1:], temp_kobe[:-1])
    plt.subplot(3, 1, 3)
    plt.scatter(temp_osaka[1:], temp_osaka[:-1])
    plt.show()


if __name__ == '__main__':
    correlate_test()
