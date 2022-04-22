import os
import extcolors
import numpy as np
import pandas as pd
from pathlib import Path
import nearest_neighbor


def listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ]


def creat_df(folder, designer1, year1) -> pd.DataFrame:
    colors_list = []
    i = 1
    for filename in os.listdir(folder):
        colors, pixel_count = extcolors.extract_from_path(folder + "\\" + filename)
        colors_list.append((colors, pixel_count))
        i += 1

    designer2 = [designer1] * len(colors_list)
    year2 = [year1] * len(colors_list)
    df2 = pd.DataFrame({"designer": designer2, "year": year2})
    i = 0
    for colors, pixel_count in colors_list:
        for color in colors:
            if str(color[0]) not in df2.columns:
                df2[str(color[0])] = 0
            df2[str(color[0])][i] = color[1] / pixel_count
        i += 1
    return df2


def create_all_data() -> pd.DataFrame:
    all_data = pd.DataFrame()
    designer_list = listdirs("data")
    for designer in designer_list:
        designer_name = Path(designer).name
        print(designer_name)
        years = listdirs(designer)
        for year in years:
            year_name = Path(year).name
            df = creat_df(year, designer_name, year_name)
            print(year_name)
            frames = [df, all_data]
            all_data = pd.concat(frames)

    all_data = all_data.reset_index()
    all_data = all_data.drop("index", 1)
    all_data = all_data.fillna(0)
    all_data.to_csv(r"C:\Users\feino\PycharmProjects\fashionProject\CHECK.csv")
    return all_data


def loss_by_designer(all_data: pd.DataFrame):
    Y = all_data["designer"]
    X = all_data.drop("designer", 1)

    # split the data into test & train
    train_X, train_y, test_X, test_y = nearest_neighbor.split_train_test(X, Y)

    # creat an estimator for nearest_neighbor
    my_estimator_by_designer = nearest_neighbor.NearestNeighbor(50)

    # overall loss by s
    loss_1 = my_estimator_by_designer._loss(train_X.to_numpy(), train_y.to_numpy())
    print("overall loss:" , str(loss_1))

    # check the loss of each designer
    for designer in ['chanel', 'dior', 'moschino', 'versace', 'Iris Van Herpen']:
        X_mid = test_X[test_X["designer"] == designer]
        y_mid = test_y[test_y["designer"] == designer]
        loss_2 = my_estimator_by_designer._loss(X_mid.to_numpy(), y_mid.to_numpy())
        print(designer + " loss:" , str(loss_2))

def loss_by_year(all_data: pd.DataFrame):
    Y = all_data["year"]
    X = all_data.drop("year", 1)

    # split the data into test & train
    train_X, train_y, test_X, test_y = nearest_neighbor.split_train_test(X, Y)

    # creat an estimator for nearest_neighbor
    my_estimator_by_designer = nearest_neighbor.NearestNeighbor(50)

    # overall loss by s
    loss_1 = my_estimator_by_designer._loss(train_X.to_numpy(), train_y.to_numpy())
    print("overall loss:" , str(loss_1))

    # check the loss of each year
    for i in range(2002, 2023):
        X_mid = test_X[test_X["year"] == str(i)]
        y_mid = test_y[test_y["year"] == str(i)]
        loss_2 = my_estimator_by_designer._loss(X_mid.to_numpy(), y_mid.to_numpy())
        print(str(i) + "loss:" , str(loss_2))


def make_color():
    all_data = create_all_data()

    # # loss by designer with the years
    # print("**************** loss by designer with the years ****************")
    # loss_by_designer(all_data)
    #
    # # loss by designer without the years
    # print("**************** loss by designer without the years ****************")
    # loss_by_designer(all_data.drop("year", 1))
    #
    # # loss by year with the designer
    # print("**************** loss by year with the designer ****************")
    # loss_by_designer(all_data)
    #
    # # loss by year without the designer
    # print("**************** loss by year without the designer ****************")
    # loss_by_designer(all_data.drop("designer", 1))