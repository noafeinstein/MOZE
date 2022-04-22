import os
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import nearest_neighbor


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


def listdirs(folder):
    return [
        d for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ]


def return_designer(name) -> int:
    if name == "chanel":
        return 0
    elif name == "dior":
        return 1
    elif name == "Iris Van Herpen":
        return 2
    elif name == "moschino":
        return 3
    elif name == "versace":
        return 4

def return_year(year) -> int:
    if year == "2022":
        return 20
    elif year == "2021":
        return 19
    elif year == "2020":
        return 18
    elif year == "2019":
        return 17
    elif year == "2018":
        return 16
    elif year == "2017":
        return 15
    elif year == "2016":
        return 14
    elif year == "2015":
        return 13
    elif year == "2014":
        return 12
    elif year == "2013":
        return 11
    elif year == "2012":
        return 10
    elif year == "2011":
        return 9
    elif year == "2010":
        return 8
    elif year == "2009":
        return 7
    elif year == "2008":
        return 6
    elif year == "2007":
        return 5
    elif year == "2006":
        return 4
    elif year == "2005":
        return 3
    elif year == "2004":
        return 2
    elif year == "2003":
        return 1
    elif year == "2002":
        return 0

def creating_data() -> pd.DataFrame:
    features = ["designer", "year"]
    for i in range(1, 785):
        features.append("pixel" + str(i))
    all_data = pd.DataFrame(columns=features)

    designer_list = listdirs("data")
    for designer in designer_list:
        designer_name = return_designer(Path(designer).name)
        print(designer_name)
        years = listdirs(designer)
        for year in years:
            year_name = return_year(Path(year).name)
            for filename in os.listdir(year):
                new_val = [designer_name, int(year_name)] + imageprepare(year + "\\" + filename)
                a_series = pd.Series(new_val, index=all_data.columns)
                all_data = all_data.append(a_series, ignore_index=True)

    all_data.to_csv(r"C:\Users\feino\PycharmProjects\fashionProject\Pixel.csv")
    return all_data


def training_model1(inp_train: pd.DataFrame, out_train : pd.Series,
                   inp_test: pd.DataFrame, out_test: pd.Series ,k: int):
    # Step 1: Training and Testing Data Split
    inp_train = inp_train.to_numpy()
    inp_test = inp_test.to_numpy()
    inp_train = np.delete(inp_train, [0], 1)
    inp_test = np.delete(inp_test, [0], 1)

    inp_train = np.reshape(inp_train, (3652, 28, 28))
    inp_test = np.reshape(inp_test, (1217, 28, 28))

    inp_train = inp_train / 255.0
    inp_test = inp_test / 255.0

    # Step 2: Building, Compiling, and Training the model
    my_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(k)
    ])
    my_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    my_model.fit(inp_train, out_train, epochs=15)
    # model = tf.keras.Sequential([my_model, tf.keras.layers.Softmax()])
    return my_model


def make_prediction(param, param1):
    prediction = []
    for i in range(len(param)):
        prediction.append(np.argmax(param[i]))
    prediction = np.array(prediction)
    compare = prediction == param1
    count = np.count_nonzero(compare)
    return (count / param1.shape[0]) * 100


def apply_minist1():
    # start_data = creating_data()
    start_data = pd.read_csv("data/labels/Pixel.csv")
    x = start_data.drop("year", 1)
    x = x.drop("designer", 1)
    y = start_data["designer"]
    inp_train, out_train, inp_test, out_test = nearest_neighbor.split_train_test(x, y)

    # making a model for label "designer"
    my_model = training_model1(inp_train, out_train, inp_test, out_test, 5)

    # checking overall accuracy
    inp_test1 = inp_test.to_numpy()
    inp_test1 = np.delete(inp_test1, [0], 1)
    inp_test1 = np.reshape(inp_test1, (1217, 28, 28))
    prediction = make_prediction(my_model.predict(inp_test1), out_test)
    print("overall prediction by designer: ", prediction, "%", "success", "num of args:", out_test.shape[0])

    # checking accuracy by designer
    inp_test2 = inp_test
    inp_test2["designer"] = out_test
    for designer in range(5):
        inp_test3 = inp_test2[inp_test2["designer"] == designer]
        out_test3 = inp_test3["designer"]
        length = out_test3.shape[0]
        inp_test3 = inp_test3.drop("designer", 1)
        inp_test3 = inp_test3.to_numpy()
        inp_test3 = np.delete(inp_test3, [0], 1)
        inp_test3 = np.reshape(inp_test3, (inp_test3.shape[0], 28, 28))

        prediction = make_prediction(my_model.predict(inp_test3), out_test3)
        print("overall prediction by designer", designer, ":", prediction, "%", "success", "num of args:", out_test3.shape[0])

    print("########################################################################################")
    y = start_data["year"]
    inp_train, out_train, inp_test, out_test = nearest_neighbor.split_train_test(x, y)

    # making a model for label "designer"
    my_model = training_model1(inp_train, out_train, inp_test, out_test, 21)

    # checking overall accuracy
    inp_test1 = inp_test.to_numpy()
    inp_test1 = np.delete(inp_test1, [0], 1)
    inp_test1 = np.reshape(inp_test1, (inp_test1.shape[0], 28, 28))
    prediction = make_prediction(my_model.predict(inp_test1), out_test)
    print("overall prediction by year: ", prediction, "%", "success", "num of args:", out_test.shape[0])

    # checking accuracy by year
    inp_test2 = inp_test
    inp_test2["year"] = out_test
    for year in range(21):
        inp_test3 = inp_test2[inp_test2["year"] == year]
        out_test3 = inp_test3["year"]
        length = out_test3.shape[0]
        inp_test3 = inp_test3.drop("year", 1)
        inp_test3 = inp_test3.to_numpy()
        inp_test3 = np.delete(inp_test3, [0], 1)
        inp_test3 = np.reshape(inp_test3, (inp_test3.shape[0], 28, 28))

        prediction = make_prediction(my_model.predict(inp_test3), out_test3)
        print("overall prediction by year", year, ":", prediction, "%", "success", "num of args:", out_test3.shape[0])


if __name__ == '__main__':
    # creating_data()
    apply_minist1()
