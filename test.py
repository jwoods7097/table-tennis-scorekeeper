import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

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
baseline_loss = pd.read_csv("runs/baseline_losses.csv")

def perform_hypothesis_test(test, tvalue, pvalue):
    print(f"t = {tvalue}, p = {pvalue}")
    if pvalue < 0.05:
        print("Reject null hypothesis for " + test)
    else:
        print("Do not reject null hypothesis for " + test)
    print()

yolo_ball_results = ball_fold_results.groupby('epoch').mean()
yolo_event_results = event_fold_results.groupby('epoch').mean()

yolo_ball_loss = yolo_ball_results['val/box_loss']
yolo_event_loss = yolo_event_results['val/loss']
baseline_ball_loss = baseline_loss['ball_loss']
baseline_event_loss = baseline_loss['event_loss']
yolo_ball_accuracy = yolo_ball_results['metrics/mAP50(B)']
yolo_event_accuracy = yolo_event_results['metrics/accuracy_top1']
# From https://arxiv.org/pdf/2004.09927.pdf tables 7 and 8, frame width of 9 was used in their training
baseline_ball_accuracy = 0.980
baseline_event_accuracy = 0.979

# Calculate means for reference
print("YOLOv8 validation box loss mean: ", yolo_ball_loss.mean())
print("YOLOv8 event loss mean: ", yolo_event_loss.mean())
print("TTNet validation box loss mean: ", baseline_ball_loss.mean())
print("TTNet event loss mean: ", baseline_event_loss.mean())
print("YOLOv8 ball accuracy mean: ", yolo_ball_accuracy.mean())
print("YOLOv8 event accuracy mean: ", yolo_event_accuracy.mean())
print()

# Perform t-tests
t, p = ttest_ind(yolo_ball_loss, baseline_ball_loss, equal_var=False, alternative='less')
perform_hypothesis_test("ball loss t-test", t, p)
t, p = ttest_ind(yolo_event_loss, baseline_event_loss, equal_var=False, alternative='less')
perform_hypothesis_test("event loss t-test", t, p)
t, p = ttest_1samp(yolo_ball_accuracy, baseline_ball_accuracy, alternative='greater')
perform_hypothesis_test("ball accuracy t-test", t, p)
t, p = ttest_1samp(yolo_event_accuracy, baseline_event_accuracy, alternative='greater')
perform_hypothesis_test("event accuracy t-test", t, p)
