import math
import os
import requests
import shutil
import tarfile

base_len = 80

current_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(current_path, "train_data")
output_path = os.path.join(train_data_path, "flowers")
input_path = os.path.join(current_path, "jpg")

all_classes = ["Daffodil", "Snowdrop", "Lily Valley", "Bluebell", "Crocus", "Iris", "Tiger Lily", "Tulip", "Fritilarry",
               "Sunflower", "Daisy", "Colts Foot", "Dandelion", "Cowslip", "Buttercup", "Windflower", "Pansy"]

extract_classes = ["Bluebell", "Buttercup", "Crocus", "Daisy", "Sunflower"]


def download_file(url):
    filename = url.split('/')[-1]
    with requests.get(url, stream=True) as req:
        with open(filename, 'wb') as file:
            shutil.copyfileobj(req.raw, file)

    return filename


def extract_file(path):
    tf = tarfile.open(path)
    tf.extractall()


def classifying_data():

    if os.path.exists(train_data_path):
        shutil.rmtree(train_data_path)

    os.mkdir(train_data_path)
    os.mkdir(output_path)

    for flower_class_name in all_classes:
        if flower_class_name in extract_classes:
            os.mkdir(os.path.join(output_path, flower_class_name))

    allfiles = os.listdir(input_path)
    file_list = [fname for fname in allfiles if fname.endswith('.jpg')]
    file_list.sort()

    for i, f in enumerate(file_list):
        try:
            if not f.endswith("txt"):
                shutil.copyfile(os.path.join(input_path, f),
                                os.path.join(output_path, all_classes[math.floor(i/base_len)], f))

        except FileNotFoundError:
            pass


def workdir_cleanup(downloaded_file):
    if os.path.exists(input_path):
        shutil.rmtree(input_path)
    if os.path.isfile(os.path.join(current_path, downloaded_file)):
        os.remove(os.path.join(current_path, downloaded_file))


def main():
    try:
        downloaded_file = download_file("http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz")
    except Exception as e:
        print("Download Error: ", e)
        return

    try:
        extract_file(os.path.join(current_path, downloaded_file))
    except Exception as e:
        print("Extract Error: ", e)
        return

    try:
        classifying_data()
    except Exception as e:
        print("Data Classifying Error: ", e)
        return

    try:
        workdir_cleanup(downloaded_file)
    except Exception as e:
        print("Working Directory Cleanup Error: ", e)
        return


if __name__ == "__main__":
    main()
    print("データを準備しました。")