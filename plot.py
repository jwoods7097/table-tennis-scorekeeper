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
event_fold_results = import_data("runs/classify/event-tracker")

if not os.path.exists("plots"):
    os.makedirs("plots")

# General results for ball model folds
# means: If true, plots an average and standard deviation for all folds instead of each fold independently
def plot_ball_results(means):
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
    plt.savefig("plots/ball_results_means.png" if means else "plots/ball_results.png")
    plt.clf()

plot_ball_results(True)
plot_ball_results(False)

# General results for event model folds
# means: If true, plots an average and standard deviation for all folds instead of each fold independently
def plot_event_results(means):
    extra = dict()
    if not means:
        extra = {"hue": "fold", "palette": "deep"}
    
    plt.figure(figsize=(9,3))

    plt.subplot(1,3,1)
    seaborn.lineplot(data=event_fold_results, x="epoch", y="train/loss", **extra)
    plt.title("train/loss")
    plt.ylabel("")

    plt.subplot(1,3,2)
    seaborn.lineplot(data=event_fold_results, x="epoch", y="val/loss", **extra)
    plt.title("val/loss")
    plt.ylabel("")

    plt.subplot(1,3,3)
    seaborn.lineplot(data=event_fold_results, x="epoch", y="metrics/accuracy_top1", **extra)
    plt.title("metrics/accuracy_top1")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig("plots/event_results_means.png" if means else "plots/event_results.png")
    plt.clf()

plot_event_results(True)
plot_event_results(False)

# training cls_loss for all 5 folds of ball model
# training ball loss from baseline
plt.figure(figsize=(3,3))
seaborn.lineplot(data=ball_fold_results, x="epoch", y="train/cls_loss")
plt.title("Training Loss")
plt.ylabel("Loss")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/ball_train_loss.png')
plt.clf()

# validation cls_loss for all 5 folds of ball model
# validation ball loss from baseline
seaborn.lineplot(data=ball_fold_results, x="epoch", y="val/cls_loss")
plt.title("Validation Loss")
plt.ylabel("Loss")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/ball_valid_loss.png')
plt.clf()

# mAP50 for all 5 folds of ball model (compare with accuracy from paper)
seaborn.lineplot(data=ball_fold_results, x="epoch", y="metrics/mAP50(B)")
plt.title("Mean Average Precision")
plt.ylabel("mAP50")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/ball_mAP50.png')
plt.clf()

# training loss for all 5 folds of event model
# training event loss from baseline
seaborn.lineplot(data=event_fold_results, x="epoch", y="train/loss")
plt.title("Training Loss")
plt.ylabel("Loss")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/event_train_loss.png')
plt.clf()

# validation loss for all 5 folds of event model
# validation event loss from baseline
seaborn.lineplot(data=event_fold_results, x="epoch", y="val/loss")
plt.title("Validation Loss")
plt.ylabel("Loss")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/event_valid_loss.png')
plt.clf()

# accuracy_top1 for all 5 folds of event model (compare with accuracy from paper)
seaborn.lineplot(data=event_fold_results, x="epoch", y="metrics/accuracy_top1")
plt.title("Top 1 Accuracy")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.xlabel("Epoch")
plt.savefig('plots/event_accuracy.png')
plt.clf()
