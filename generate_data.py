import os
import json
import re
import shutil

games = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'test_1', 'test_2']

for game in games:
    game_type, game_id = game.split('_')
    with open(f'D:\\Downloads\\{game}\\ball_markup.json', 'r') as json_file:
        data = json.load(json_file)
        frame_nums = [f for f in data.keys() if data[f]['x'] != -1 and data[f]['y'] != -1][:(400 if game_type == 'game' else 250)]
        for file in os.listdir('D:\\Dropbox\\Apps\\cis530\\all-frames'):
            if not (file.endswith('.py') or file.endswith('.ps1')):
                type, id, frame, ext = re.split('\_|\.', file)
                if type == game_type and id == game_id and frame in frame_nums:
                    print("Copying", file)
                    shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, "D:\\Dropbox\\Apps\\cis530\\ball-data\\" + ("train" if game_type == 'game' else "valid") + f"\\images\\{type}_{id}_{frame}.jpg")
                    x_center = data[frame]['x'] / 1920
                    y_center = data[frame]['y'] / 1080
                    width = 24 / 1920
                    height = 24 / 1080
                    with open("D:\\Dropbox\\Apps\\cis530\\ball-data\\" + ("train" if game_type == 'game' else "valid") + f"\\labels\\{type}_{id}_{frame}.txt", "w") as yolo_file:
                        yolo_file.write(f"0 {x_center} {y_center} {width} {height}")