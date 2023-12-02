import os
import json
import re
import shutil
import numpy as np

games = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'test_1', 'test_2']
file_dict = {"bounce": [], "empty_event": [], "net": []}

# Get labeled frames  
print("Sorting frames by class...")          
for game in games:
    game_type, game_id = game.split('_')
    with open(f'D:\\Downloads\\{game}\\events_markup.json', 'r') as json_file:
        data = json.load(json_file)
        frame_nums = list(data.keys())
        for file in os.listdir('D:\\Dropbox\\Apps\\cis530\\all-frames'):
            if not (file.endswith('.py') or file.endswith('.ps1')):
                type, id, frame, ext = re.split('\_|\.', file)
                if type == game_type and id == game_id and frame in frame_nums:
                    cls = data[frame]
                    file_dict[cls].append(file)

# Create folds from labeled frames
print("Generating folds...")
np.random.seed(0)                
for cls, ls in file_dict.items():
    data = np.random.choice(ls, (5, 200), False)
    for i in range(5):
        valid = data[i]
        path = f"D:\\Dropbox\\Apps\\cis530\\event-data\\fold{i}\\val\\{cls}"
        if not os.path.exists(path):
            os.makedirs(path)
        for file in valid:
            shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, f"{path}\\{file}")
            print(f"Copying {file} into validation set for fold {i}")
        
        train = np.delete(data, i, 0).flatten()
        path = f"D:\\Dropbox\\Apps\\cis530\\event-data\\fold{i}\\train\\{cls}"
        if not os.path.exists(path):
            os.makedirs(path)
        for file in train:
            shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, f"{path}\\{file}")
            print(f"Copying {file} into train set for fold {i}")
           
        # Since yolov8 classify demands a test set to work but doesn't use it
        path = f"D:\\Dropbox\\Apps\\cis530\\event-data\\fold{i}\\test\\{cls}"
        if not os.path.exists(path):
            os.makedirs(path)
        