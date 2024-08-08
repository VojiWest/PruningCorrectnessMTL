import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib.cm as cm
import argparse

parset = argparse.ArgumentParser()
parset.add_argument('--graphs', type=bool, default=False)
args = parset.parse_args()


def plot_results(avgs):
    if avgs.ndim > 1:
        colors = cm.tab20(np.linspace(0, 1, len(avgs)))
        for avg, color in zip(avgs, colors):
            plt.plot([20, 40, 60, 80, 100], avg, marker='o', markersize=5, color=color)
    else:
        plt.plot([20, 40, 60, 80, 100], avgs, marker='o', markersize=5)


def plot_all(all_data):
    found = False
    seg_classes = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"]
    for class_num in range(13):
        x = []
        y = []
        y_avg = np.zeros(5)
        count_nan = 0
        count_val = 0
        for data in all_data:
            if np.isnan(data[0]) == False and data[2] == class_num:
                x.append(data[1])
                y.append(data[0])
                y_avg[int(data[1]/20) - 1] += data[0]
                count_val += 1
            elif np.isnan(data[0]) == True and data[2] == class_num:
                x.append(data[1])
                y.append(0)
                count_nan += 1
        for i in range(5):
            y_avg[i] /= count_val + count_nan
        print("Number of NAN Values: ", count_nan, " --- Number of Valid Values: ", count_val)

        corr, pval = spearmanr(x, y)
        print("Correlation: ", corr, " --- P-Value: ", pval, " --- Class: ", seg_classes[class_num])
        print("")

        plt.plot([20, 40, 60, 80, 100], y_avg, marker='o', markersize=5)

        plt.scatter(x, y, marker='o', s=1)
        plt.xlabel('Percent Kept')
        plt.ylabel('IoU')
        plt.title('Semantic Segmentation - Base Model - Class ' + seg_classes[class_num])
        plt.show()

def get_class_cors(all_data, task, model):
    seg_classes = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"]
    depth_classes = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    sur_norm_classes = ["-x-y-z", "-x+y-z", "+x-y-z", "+x+y-z", "-x-y+z", "-x+y+z", "+x-y+z", "+x+y+z"]
    class_num_dict = { "SemSeg": 13, "Depth": 5, "SurNorm": 8 }
    for class_num in range(class_num_dict[task]):
        x = []
        y = []
        y_avg = np.zeros(5)
        count_nan = 0
        count_val = 0
        for data in all_data:
            if np.isnan(data[0]) == False and data[2] == class_num:
                x.append(data[1])
                y.append(data[0])
                y_avg[int(data[1]/20) - 1] += data[0]
                count_val += 1
            elif np.isnan(data[0]) == True and data[2] == class_num:
                if task == "SemSeg":
                    x.append(data[1])
                    y.append(0)
                count_nan += 1

        corr, pval = spearmanr(x, y)
        if task == "SemSeg":
            print(model[:-4], " :: Class: ", seg_classes[class_num], " -- Correlation: ", corr, " -- P-Value: ", pval, "     (Nan/Val: ", count_nan, "/", count_val, " -- ", count_nan + count_val, ")")
        elif task == "Depth":
            print(model[:-4], " :: Class: ", depth_classes[class_num], " -- Correlation: ", corr, " -- P-Value: ", pval, "     (Nan/Val: ", count_nan, "/", count_val, " -- ", count_nan + count_val, ")")
        elif task == "SurNorm":
            print(model[:-4], " :: Class: ", sur_norm_classes[class_num], " -- Correlation: ", corr, " -- P-Value: ", pval, "     (Nan/Val: ", count_nan, "/", count_val, " -- ", count_nan + count_val, ")")


def get_ovr_cors(all_data, task, model):
    x = []
    y = []
    y_avg = np.zeros(5)
    count_nan = 0
    count_val = 0
    for data in all_data:
        if np.isnan(data[0]) == False:
            x.append(data[1])
            y.append(data[0])
            y_avg[int(data[1]/20) - 1] += data[0]
            count_val += 1
        elif np.isnan(data[0]) == True:
            if task == "SemSeg":
                x.append(data[1])
                y.append(0)
            count_nan += 1

    corr, pval = spearmanr(x, y)
    print(model[:-4], " :: Correlation: ", corr, " -- P-Value: ", pval, "     (Nan/Val: ", count_nan, "/", count_val, " -- ", count_nan + count_val, ")")

    return corr


def show_plot(metric, task, name, model):
    plt.xlabel('Percent Kept')
    plt.ylabel(metric)
    # plt.axes().set_facecolor('#f8f6ed')
    if name == "Classes":
        if task == "SemSeg":
            plt.legend(["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"])
        if task == "Depth":
            plt.legend(["Closest", "Close", "Middle", "Far", "Farthest"])
        if task == "SurNorm":
            plt.legend(["-x-y-z", "-x+y-z", "+x-y-z", "+x+y-z", "-x-y+z", "-x+y+z", "+x-y+z", "+x+y+z"])
    else:   
        plt.legend(["Base", "50% Pruned", "75% Pruned", "87.5% Pruned", "93.75% Pruned"])
    if task == "SemSeg":
        task = "Semantic Segmentation"
        plt.title(task + " - " + model)
    elif task == "Depth":
        task = "Depth Estimation"
        plt.title(task + " - " + model)
    elif task == "SurNorm":
        task = "Surface Normal Estimation"
        plt.title(task + " - " + model)
    if name == "Overall":
        plt.savefig('Plots/Model_Plots/' + model + ' ' + task + ' -- ' + metric + 'colored.png')
    elif name == "Classes":
        plt.savefig('Plots/Class_Plots_New/' + model + ' ' + name + ' - ' + task + ' -- ' + metric + 'colored.png')
    plt.show()
    

def get_avg(data, task, image = 0):
    seg_totals = np.zeros(5, dtype=np.float64)
    seg_counts = np.zeros(5)
    extra_seg_totals = np.zeros(5, dtype=np.float64)
    extra_seg_counts = np.zeros(5)
    for row in data.iterrows():
        if row[1]['Task'] == task: # and row[1]['Image Number'] == image:
            idx = int(row[1]['ROAD Percentile']/20) - 1
            if np.isnan(row[1]['Class Metric']) == False:
                seg_totals[idx] += row[1]['Class Metric']
                seg_counts[idx] += 1

                extra_seg_totals[idx] += row[1]['Extra Class Metric']
                extra_seg_counts[idx] += 1
            elif np.isnan(row[1]['Class Metric']) == True:
                if task == "SemSeg":
                    seg_totals[idx] += 0
                    seg_counts[idx] += 1

                extra_seg_totals[idx] += row[1]['Extra Class Metric']
                extra_seg_counts[idx] += 1

    seg_avgs = np.zeros(5)
    extra_avgs = np.zeros(5)
    for i in range(5):
        seg_avgs[i] = seg_totals[i] / seg_counts[i]
        extra_avgs[i] = extra_seg_totals[i] / extra_seg_counts[i]

    return seg_avgs, extra_avgs

def get_per_class_avgs(data, task, image = 0):
    # task class dictionary
    task_class = { "SemSeg": 13, "Depth": 5, "SurNorm": 8 }
    class_seg_totals = np.zeros((task_class[task], 5), dtype=np.float64)
    class_seg_counts = np.zeros((task_class[task], 5))
    extra_metric_totals = np.zeros((task_class[task], 5), dtype=np.float64)
    extra_metric_counts = np.zeros((task_class[task], 5))

    for row in data.iterrows():
        if row[1]['Task'] == task:
            idx = int(row[1]['ROAD Percentile']/20) - 1
            if np.isnan(row[1]['Class Metric']) == False: # and row[1]['Image Number'] == image:
                class_num = int(row[1]['Class'])
                class_seg_totals[class_num][idx] += row[1]['Class Metric']
                class_seg_counts[class_num][idx] += 1

                extra_metric_totals[class_num][idx] += row[1]['Extra Class Metric']
                extra_metric_counts[class_num][idx] += 1
            elif np.isnan(row[1]['Class Metric']) == True:
                class_num = int(row[1]['Class'])
                if task == "SemSeg":
                    class_seg_totals[class_num][idx] += 0
                    class_seg_counts[class_num][idx] += 1

                extra_metric_totals[class_num][idx] += row[1]['Extra Class Metric']
                extra_metric_counts[class_num][idx] += 1

    class_seg_avgs = np.zeros((task_class[task], 5))
    extra_class_avgs = np.zeros((task_class[task], 5))
    for i in range(task_class[task]):
        for j in range(5):
            if class_seg_counts[i][j] != 0:
                class_seg_avgs[i][j] = class_seg_totals[i][j] / class_seg_counts[i][j]
            if extra_metric_counts[i][j] != 0:
                extra_class_avgs[i][j] = extra_metric_totals[i][j] / extra_metric_counts[i][j]
    
    ### Remove rows with all 0 values as they have no data ###
    remove_rows = []
    for idx, class_row in enumerate(class_seg_avgs):
        if sum(class_row) == 0:
            remove_rows.append(idx)
    for idx in remove_rows:
        class_seg_avgs = np.delete(class_seg_avgs, idx, axis=0)
        for i in range(len(remove_rows)):
            remove_rows[i] -= 1


    return class_seg_avgs, extra_class_avgs

def get_all(data, task):
    # Get list of all metrics for each percentile of task
    seg_all = []
    for row in data.iterrows():
        if row[1]['Task'] == task:
            seg_all.append((row[1]['Class Metric'], row[1]['ROAD Percentile'], row[1]['Class']))
            # if np.isnan(row[1]['Class Metric']):
            #     print("NAN Value Found", row[1]['Class Metric'])

    return seg_all

def plot_box(data, task, model):
    seg_totals = [[], [], [], [], []]

    for row in data.iterrows():
        if row[1]['Task'] == task: # and row[1]['Image Number'] == image:
            idx = int(row[1]['ROAD Percentile']/20) - 1
            if np.isnan(row[1]['Class Metric']) == False:
                seg_totals[idx].append(row[1]['Class Metric'])
            elif np.isnan(row[1]['Class Metric']) == True:
                if task == "SemSeg":
                    seg_totals[idx].append(0)

    plt.boxplot(seg_totals, showfliers=False)
    plt.xlabel('Percent Kept')
    if task == "SemSeg":
        task = "Semantic Segmentation"
        plt.title(task + " - " + model[:-4])
        plt.ylabel('IoU')
    elif task == "Depth":
        task = "Depth Estimation"
        plt.title(task + " - " + model[:-4])
        plt.ylabel('Relative Error')
    elif task == "SurNorm":
        task = "Surface Normal Estimation"
        plt.title(task + " - " + model[:-4])
        plt.ylabel('Mean')


def get_results():
    ratio_groups = [["Base_1_F", "Base_2_F", "Base_3_F"], ["Pruned50_1_F", "Pruned50_2_F", "Pruned50_3_F"], ["Pruned75_1_F", "Pruned75_2_F", "Pruned75_3_F"], ["Pruned87_5_1_F", "Pruned87_5_2_F", "Pruned87_5_3_F"], ["Pruned93_75_1_F", "Pruned93_75_2_F", "Pruned93_75_3_F"]]
    tasks = ["SemSeg", "Depth", "SurNorm"]
    metrics = ["IoU", "Relative Error", "Mean"]

    ### Plot average results for each pruning ratio ###
    if args.graphs:
        print(" --- Plotting Average Results for Each Pruning Ratio --- ")
        for task in tasks:
            plt.figure(facecolor='#f8f6ed')
            plt.axes().set_facecolor('#f8f6ed')
            for models in ratio_groups:
                ratio_data = []
                for model in models:
                    data = pd.read_csv('ROAD_Scores/Final/ROAD_Results_' + model + '.csv')
                    ratio_data.append(data)
                all_data = pd.concat(ratio_data)
                task_avgs, xtra_avgs = get_avg(all_data, task)
                metric = metrics[tasks.index(task)]
                plot_results(task_avgs)
                #   # plot_results(xtra_avgs)
            show_plot(metric, task, "Overall", "Average")
        plt.close()

        ### Plot average results for each class for each pruning ratio ###
        print(" --- Plotting Average Results for Each Class for Each Pruning Ratio ---")
        for task in tasks:
            for models in ratio_groups:
                ratio_data = []
                for model in models:
                    data = pd.read_csv('ROAD_Scores/Final/ROAD_Results_' + model + '.csv')
                    ratio_data.append(data)
                all_data = pd.concat(ratio_data)
                task_class_avgs, xtra_class_avg = get_per_class_avgs(all_data, task)
                metric = metrics[tasks.index(task)]
                plot_results(task_class_avgs)
                show_plot(metric, task, "Classes", model[:-4])

    ### Plot box plots for each pruning ratio ###
    print(" --- Plotting Box Plots for Each Pruning Ratio --- ")
    for task in tasks:
        for models in ratio_groups:
            ratio_data = []
            for model in models:
                data = pd.read_csv('ROAD_Scores/Final/ROAD_Results_' + model + '.csv')
                ratio_data.append(data)
            all_data = pd.concat(ratio_data)
            plot_box(all_data, task, model)
        plt.show()

    ### Get correlation values for each class at each pruning ratio ###
    for task in tasks:
        print("Task: ", task)
        # for image in range(100):
        for models in ratio_groups:
            ratio_data = []
            for model in models:
                data = pd.read_csv('ROAD_Scores/Final/ROAD_Results_' + model + '.csv')
                ratio_data.append(data)
            all_data = pd.concat(ratio_data)
            task_all = get_all(all_data, task)
            metric = metrics[tasks.index(task)]
            get_class_cors(task_all, task, model)
            print("")
        print("--------------------")

    ### Get correlation values for each pruning ratio ###
    model_avgs = np.zeros((len(ratio_groups), 3))
    for task in tasks:
        print("Task: ", task)
        # for image in range(100):
        for models in ratio_groups:
            ratio_data = []
            for model in models:
                data = pd.read_csv('ROAD_Scores/Final/ROAD_Results_' + model + '.csv')
                ratio_data.append(data)
            all_data = pd.concat(ratio_data)
            task_all = get_all(all_data, task)
            metric = metrics[tasks.index(task)]
            corr = get_ovr_cors(task_all, task, model)
            if task == "SemSeg":
                model_avgs[ratio_groups.index(models)][0] = corr
            elif task == "Depth":
                model_avgs[ratio_groups.index(models)][1] = corr
            elif task == "SurNorm":
                model_avgs[ratio_groups.index(models)][2] = corr
        print("--------------------")

    for model in ratio_groups:
        seg = model_avgs[ratio_groups.index(model)][0]
        depth = model_avgs[ratio_groups.index(model)][1]
        sur_norm = model_avgs[ratio_groups.index(model)][2]
        print(model[0][:-4], "model :: Average Correlation: ", (abs(seg) + abs(depth) + abs(sur_norm)) / 3)
    

def main():
    get_results()

if __name__ == '__main__':
    main()