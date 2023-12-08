import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp, shapiro, levene, boxcox

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
baseline_loss = pd.read_csv("losses.csv")

def perform_hypothesis_test(test, pvalue):
    print(f"p = {pvalue}")
    if pvalue < 0.05:
        print("Reject null hypothesis for " + test)
    else:
        print("Do not reject null hypothesis for " + test)

yolo_ball_results = ball_fold_results.groupby('epoch').mean()
yolo_event_results = event_fold_results.groupby('epoch').mean()

yolo_ball_loss = yolo_ball_results['val/cls_loss']
yolo_event_loss = yolo_event_results['val/loss']
baseline_ball_loss = baseline_loss['ball_loss']
baseline_event_loss = baseline_loss['event_loss']
yolo_ball_accuracy = yolo_ball_results['metrics/mAP50(B)']
yolo_event_accuracy = yolo_event_results['metrics/accuracy_top1']
# From https://arxiv.org/pdf/2004.09927.pdf tables 7 and 8, frame width of 9 was used
baseline_ball_accuracy = 0.980
baseline_event_accuracy = 0.979

# Check if error is normally distributed
_, p = shapiro(yolo_ball_loss)
perform_hypothesis_test("yolo ball shapiro test", p)
_, p = shapiro(yolo_event_loss)
perform_hypothesis_test("yolo event shapiro test", p)
_, p = shapiro(baseline_ball_loss)
perform_hypothesis_test("baseline ball shapiro test", p)
_, p = shapiro(baseline_event_loss)
perform_hypothesis_test("baseline event shapiro test", p)

# Check if variance is equal
_, p = levene(yolo_ball_loss, baseline_ball_loss)
perform_hypothesis_test("ball levene test", p)
_, p = levene(yolo_event_loss, baseline_event_loss)
perform_hypothesis_test("event levene test", p)

# Perform t-tests
_, p = ttest_ind(yolo_ball_loss, baseline_ball_loss, equal_var=False, alternative='less')
perform_hypothesis_test("ball loss t-test", p)
_, p = ttest_ind(yolo_event_loss, baseline_event_loss, equal_var=False, alternative='less')
perform_hypothesis_test("event loss t-test", p)
_, p = ttest_1samp(yolo_ball_accuracy, baseline_ball_accuracy, alternative='greater')
perform_hypothesis_test("ball accuracy t-test", p)
_, p = ttest_1samp(yolo_event_accuracy, baseline_event_accuracy, alternative='greater')
perform_hypothesis_test("event accuracy t-test", p)
