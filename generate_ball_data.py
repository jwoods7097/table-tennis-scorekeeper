import os
import json
import re
import shutil
import numpy as np
import random

games = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5']
ball_frames = dict()

# Get labeled frames
print("Gathering labeled frames...")
for game in games:
    game_type, game_id = game.split('_')
    with open(f'D:\\Downloads\\{game}\\ball_markup.json', 'r') as json_file:
        data = json.load(json_file)
        frame_nums = [f for f in data.keys() if data[f]['x'] != -1 and data[f]['y'] != -1]
        for file in os.listdir('D:\\Dropbox\\Apps\\cis530\\all-frames'):
            if not (file.endswith('.py') or file.endswith('.ps1')):
                type, id, frame, ext = re.split('\_|\.', file)
                if type == game_type and id == game_id and frame in frame_nums:
                    # Create label
                    x_center = data[frame]['x'] / 1920
                    y_center = data[frame]['y'] / 1080
                    width = 24 / 1920
                    height = 24 / 1080
                    ball_frames[file] = f"0 {x_center} {y_center} {width} {height}"
                    
# Generate folds   
print("Generating folds...")
np.random.seed(0)               
data = np.random.choice(list(ball_frames.keys()), (5, 1000), False)
for i in range(5):
    valid = data[i]
    path = f"D:\\Dropbox\\Apps\\cis530\\ball-data\\fold{i}\\valid"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "\\images")
        os.makedirs(path + "\\labels")
    for file in valid:
        shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, f"{path}\\images\\{file}")
        with open(path + f"\\labels\\{file[:-4]}.txt", "w") as yolo_file:
            yolo_file.write(ball_frames[file])
        print(f"Copying {file} into validation set for fold {i}")
    
    train = np.delete(data, i, 0).flatten()
    path = f"D:\\Dropbox\\Apps\\cis530\\ball-data\\fold{i}\\train"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "\\images")
        os.makedirs(path + "\\labels")
    for file in train:
        shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, f"{path}\\images\\{file}")
        with open(path + f"\\labels\\{file[:-4]}.txt", "w") as yolo_file:
            yolo_file.write(ball_frames[file])
        print(f"Copying {file} into training set for fold {i}")
