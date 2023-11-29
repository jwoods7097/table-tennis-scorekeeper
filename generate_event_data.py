import os
import json
import re
import shutil

games = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5', 'test_1', 'test_2']

for game in games:
    game_type, game_id = game.split('_')
    with open(f'D:\\Downloads\\{game}\\events_markup.json', 'r') as json_file:
        data = json.load(json_file)
        frame_nums = list(data.keys())
        for file in os.listdir('D:\\Dropbox\\Apps\\cis530\\all-frames'):
            if not (file.endswith('.py') or file.endswith('.ps1')):
                type, id, frame, ext = re.split('\_|\.', file)
                if type == game_type and id == game_id and frame in frame_nums:
                    print("Copying", file)
                    cls = data[frame]
                    shutil.copyfile('D:\\Dropbox\\Apps\\cis530\\all-frames\\' + file, "D:\\Dropbox\\Apps\\cis530\\event-data\\" + ("train" if game_type == 'game' else "valid") + f"\\{cls}\\{type}_{id}_{frame}.jpg")