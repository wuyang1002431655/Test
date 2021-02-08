
import os
import shutil


def change_image():
    root = '../dataset'
    if 'DogsVSCats' in os.listdir(root):
        print("DogsVSCats已存在")
        return
    os.makedirs(os.path.join(root, 'DogsVSCats/train/cat'))
    os.makedirs(os.path.join(root, 'DogsVSCats/train/dog'))
    os.makedirs(os.path.join(root, 'DogsVSCats/valid/cat'))
    os.makedirs(os.path.join(root, 'DogsVSCats/valid/dog'))
    # 猫狗训练集各有12500张
    path = os.path.join(root, 'DogCat/train')
    num_cat = 0
    num_dog = 0
    for i in os.listdir(path):
        if 'dog' in i:
            if num_dog < 10000:
                shutil.copy(os.path.join(path, i), os.path.join(root, 'DogsVSCats/train/dog'))
                num_dog = num_dog + 1
            else:
                shutil.copy(os.path.join(path, i), os.path.join(root, 'DogsVSCats/valid/dog'))
        elif 'cat' in i:
            if num_cat<10000:
                shutil.copy(os.path.join(path,i),os.path.join(root, 'DogsVSCats/train/cat'))
                num_cat=num_cat+1
            else:
                shutil.copy(os.path.join(path,i),os.path.join(root, 'DogsVSCats/valid/cat'))
    print("处理完毕")


change_image()