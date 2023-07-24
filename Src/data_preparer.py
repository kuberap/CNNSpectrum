from os import listdir
from os.path import isfile, join
import numpy as np

def load_labeled_data(paths: dict, parse_file_function, maxdata_count=None):
    titles = []
    index = 0
    X = []
    y = []
    for path, label in paths.items(): #cti pres vsechny definovane adresare
        for file in listdir(path):
            f_path = join(path, file)
            if isfile(f_path):
                x_raw = np.load(f_path,allow_pickle=True) # nacti cely soubor
                data, title =parse_file_function(x_raw) # vyzjisti potrebna data + titulek
                y.append(label)
                X.append(data) # pripoj si pole dat
                titles.append(title)
                index = index + 1
            if maxdata_count is not None and index > maxdata_count:
               break
    return np.vstack(X), np.array(y), titles



def roll_X_data(X_data, y_data, c1_shift, c0_shift, value=0):
    """
    Provede rotaci testovaich dat, kazdou tridu rotuje jinak.
    Chybejici - vyrotovane hodnoty jsou doplneny value
    :param X_data:
    :param y_data:
    :param c1_shift:
    :param c0_shift:
    :return:
    """
    data = []
    with np.printoptions(edgeitems=4, precision=3, suppress=True):
        for index in range(len(X_data)):  # pro kazdy radek matice
            if y_data[index] == 1:  # kdyz je to trida 1
                rotated = np.roll(X_data[index], c1_shift)  # zrotuj
                if c1_shift >= 0:  # rotace v pravo
                    rotated[0:c1_shift] = value  # prepisuji zleva
                else:
                    rotated[c1_shift:] = value  # prepisuji zprava
            else:  # trida 0
                rotated = np.roll(X_data[index], c0_shift)  # zrotuj
                if c0_shift >= 0:  # rotace v pravo
                    rotated[0:c0_shift] = value  # prepisuji zleva
                else:
                    rotated[c0_shift:] = value  # prepisuji zprava
            # print(f"O:{X_data[index]}")
            # print(f"R:{rotated}")
            # print("-"*30)
            data.append(rotated)
    return np.array(data)


def random_rotate_dataset(X_data,y_data, left = -10, right = 10, steps = 5, value = 0):
    data = []
    labels = []
    for step in range(steps): # opakuj dany pocetkrat
        for index in range(len(X_data)):  # pro kazdy radek matice
            rot = np.random.randint(left, right)
            rotated = np.roll(X_data[index], rot)
            if rot >= 0:  # rotace v pravo
                rotated[0:rot] = value  # prepisuji zleva
            else:
                rotated[rot:] = value  # prepisuji zprava
            data.append(rotated)
            labels.append(y_data[index])

    return np.array(data), np.array(labels)
