import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
import os

# Load in results
def import_data(path, debug=False):
    dfs = []
    for i in range(5):
        df = pd.read_csv(f"{path}-fold{i}/results.csv")
        df['fold'] = [i for j in range(100)]
        dfs.append(df)
    data = pd.concat(dfs)
    data.rename(columns=(lambda x: x.strip()), inplace=True)
    if debug:
        data.to_excel(path[path.rfind('/') + 1:] + ".xlsx")
    return data

ball_fold_results = import_data("runs/detect/ball-tracker-10k")

if not os.path.exists("plots"):
    os.makedirs("plots")

# general results for ball model folds
def plot_results(means):
    extra = dict()
    if not means:
        extra = {"hue": "fold", "palette": "deep"}
    
    plt.figure(figsize=(15,6))

    plt.subplot(2,5,1)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="train/box_loss", **extra)
    plt.title("train/box_loss")
    plt.ylabel("")

    plt.subplot(2,5,2)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="train/cls_loss", **extra)
    plt.title("train/cls_loss")
    plt.ylabel("")

    plt.subplot(2,5,3)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="train/dfl_loss", **extra)
    plt.title("train/dfl_loss")
    plt.ylabel("")

    plt.subplot(2,5,4)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="metrics/precision(B)", **extra)
    plt.title("metrics/precision(B)")
    plt.ylabel("")

    plt.subplot(2,5,5)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="metrics/recall(B)", **extra)
    plt.title("metrics/recall(B)")
    plt.ylabel("")

    plt.subplot(2,5,6)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="val/box_loss", **extra)
    plt.title("val/box_loss")
    plt.ylabel("")

    plt.subplot(2,5,7)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="val/cls_loss", **extra)
    plt.title("val/cls_loss")
    plt.ylabel("")

    plt.subplot(2,5,8)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="val/dfl_loss", **extra)
    plt.title("val/dfl_loss")
    plt.ylabel("")

    plt.subplot(2,5,9)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="metrics/mAP50(B)", **extra)
    plt.title("metrics/mAP50(B)")
    plt.ylabel("")

    plt.subplot(2,5,10)
    seaborn.lineplot(data=ball_fold_results, x="epoch", y="metrics/mAP50-95(B)", **extra)
    plt.title("metrics/mAP50-95(B)")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig("plots/results_means.png" if means else "plots/results.png")
    plt.clf()

plot_results(True)
plot_results(False)

# training cls_loss for all 5 folds of ball model
# training ball loss from baseline
plt.figure(figsize=(6.4,4.8))
seaborn.lineplot(data=ball_fold_results, x="epoch", y="train/cls_loss")
plt.title("Training Loss")
plt.ylabel("Loss")
plt.ylim([0,1])
plt.xlabel("Training Epoch")
plt.savefig('plots/train_loss.png')

# validation cls_loss for all 5 folds of ball model
# validation ball loss from baseline

# mAP50 for all 5 folds of ball model (compare with accuracy from paper)

# training loss for all 5 folds of event model
# training event loss from baseline

# validation loss for all 5 folds of event model
# validation event loss from baseline

# accuracy_top1 for all 5 folds of event model (compare with accuracy from paper)
