import csv
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
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 4:  # rain 4,22,40
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 8:  # air pressure 8,26,44
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 11:  # wind 11,29,47
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 13:  # wind direction 13,31,49
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
            if i % 18 == 16:  # temperature 16,34,52
                if i / 18 == 0:
                    kyoto_temp.append(data_unit[i])
                if i / 18 == 1:
                    osaka_temp.append(data_unit[i])
                if i / 18 == 2:
                    kobe_temp.append(data_unit[i])
        kyoto.append(kyoto_temp)
        osaka.append(osaka_temp)
        kobe.append(kobe_temp)
    return kyoto, osaka, kobe  # cloud,rain,pressure,wind,wd,temperature


def output_data():
    datafile = 'data11w.csv'
    temp_data = loadcsv(datafile)
    kyoto, osaka, kobe = reshape_data(temp_data)
    return kyoto, osaka, kobe


if __name__ == '__main__':
    kyoto, osaka, kobe = output_data()
    kyoto_temp = []
    for i in range(len(kyoto)):
        kyoto_temp.append(float(kyoto[i][5]))
    xs = []
    ys = []
    for(i, val) in enumerate(kyoto_temp):
        xs.append(i)
        ys.append(val)
    plt.plot(xs, ys)
    plt.show()
