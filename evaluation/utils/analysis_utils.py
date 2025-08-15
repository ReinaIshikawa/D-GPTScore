
import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
del matplotlib.font_manager.weight_dict['roman'] 
from scipy import stats
from scipy.stats import rankdata
import seaborn as sns

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

evaluation_aspects = {
    'score1': 'Subject Type',
    'score2': 'Quantity',
    'score3': 'Sbject & Camera Positioning',
    'score4': 'Size & Scale',
    'score5': 'Color',
    'score6': 'Subject Completeness',
    'score7': 'Proportions & Body Consistency',
    'score8': 'Actions & Expressions',
    'score9': 'Facial Similarity & Features',
    'score10': 'Clothing & Attributes',
    'score11': 'Surroundings',
    'score12': 'Human & Animal Interactions',
    'score13': 'Object Interactions',
    'score14': 'Subject Deformation',
    'score15': 'Surroundings Deformation',
    'score16': 'Local Artifacts',
    'score17': 'Detail & Sharpness',
    'score18': 'Style Consistency'
}

def draw_learning_curve(train_scores,  valid_scores, train_sizes, reg_dir, file_name):
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std  = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    
    plt.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
    plt.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')
    
    best_score = valid_center[len(valid_center) - 1]
    plt.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={best_score}',color='black', verticalalignment='top', horizontalalignment='right')
    
    plt.xlabel('training examples') 
    plt.ylabel('rmse') 
    plt.legend(loc='lower right') 
    plt.savefig(os.path.join(reg_dir, file_name)) 
    plt.close()

def plot_feature_importance(reg_dir, importance, file_name, linear = False, existing_methods=[]):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    for i, method in enumerate(existing_methods):
        evaluation_aspects[f"score{18+i+1}"] = method

    print(evaluation_aspects)
    labels = [evaluation_aspects[f"score{i}"] for i in range(1,19+len(existing_methods))]
    if linear:
        vals = importance
        print(vals)
    else:
        vals = [importance.get(f"score{i}", 0) for i in range(1,19+len(existing_methods))]
    print(len(labels), len(vals))
    labels.reverse()
    vals.reverse()
    plt.barh(labels, vals, color='lightblue')
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlabel('Weight')
    plt.ylabel('Evaluation Aspect')
    plt.tight_layout()
    plt.savefig(os.path.join(reg_dir, file_name)) 
    plt.close()



def save_best_model(best_model, reg_dir):
    best_model.get_booster().save_model(os.path.join(reg_dir, 'best_model.json'))


def calc_pearson_corr(x, y):
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    return correlation_coefficient

def calc_spearman_corr(x, y):
    correlation_coefficient, _ = stats.spearmanr(x, y)
    return correlation_coefficient

def calc_rank(data):
    # data: list of numpy array
    data_stack = np.vstack(data)
    ranks = rankdata(data_stack, axis=0, method="max")
    ranks_desc = len(data_stack) + 1 - ranks
    rank_list = [ranks_desc[i] for i in range(len(ranks_desc))]
    return rank_list

def scatter_plot(pred_list, user_score_list, label_list=[], reg_dir="", file_name=""):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams.update({'font.size': 18})
    if isinstance(pred_list, list):
        for pred, user_score, label in zip(pred_list, user_score_list, label_list):
            plt.scatter(user_score, pred, label=label, marker='o', s=3)
        plt.legend()
    else:
        plt.scatter(user_score_list, pred_list, marker='o', s=3, c='c')
        slope, intercept = np.polyfit(user_score_list, pred_list, 1)
        y_fit = slope * user_score_list + intercept
        plt.plot(user_score_list, y_fit, color='red', label='Regression Line')
    
    
    plt.xticks(np.arange(1, 11, 1))
    plt.yticks(np.arange(1, 11, 1))
    plt.xlabel('Human Preference Score')
    plt.ylabel('Predicted Score')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(reg_dir, file_name),bbox_inches="tight") 
    plt.close()

def save_learning_curve(evals_result, reg_dir, file_name):
    plt.plot(evals_result['train']['rmse'], label='train rmse')
    plt.plot(evals_result['eval']['rmse'], label='eval rmse')
    plt.grid()
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('rmse')
    plt.savefig(os.path.join(reg_dir, file_name)) 
    plt.close()

def ablation_bar_plot(json_data, reg_dir):
    label_list = ["default"] + [f"score{ea}" for ea in range(1,19)]
    data_list = {}
    for corr_type in ["pearson", "spearman"]:
        data_list[corr_type] = []
        for ablation_target in label_list:
            if corr_type=="pearson":
                corr = json_data[ablation_target]["total_pearson_corr"]
            else:
                corr = json_data[ablation_target]["total_spearman_corr"]
            data_list[corr_type].append(corr)
    print("data_list:", data_list)
    x = np.arange(len(label_list))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    default_value = json_data["default"]["total_pearson_corr"]
    ax.axhline(y=default_value, color='blue', linestyle='--', label="Pearson")
    default_value = json_data["default"]["total_spearman_corr"]
    ax.axhline(y=default_value, color='red', linestyle='--', label="Spearman")

    for attribute, measurement in data_list.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.4f')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation')
    ax.set_xticks(x + width, label_list)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0.65, 0.80)

    plt.savefig(os.path.join(reg_dir, "corr_ablation_bar.pdf"))
    plt.close()

def corr_matrix_plot(metric_df, reg_dir):
    correlation_matrix = metric_df.corr()
    print(correlation_matrix)
    sns.set_context("notebook", font_scale=0.7) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(reg_dir, "corr_matrix.pdf"))
    plt.close()

def comparison_plot(
    linear_test_pred_list, 
    average_test_pred_list, 
    reg_dir
    ):
    
    linear_test_pred_list = [round(num, 2) for num in linear_test_pred_list]
    average_test_pred_list = [round(num, 2) for num in average_test_pred_list]
    print(linear_test_pred_list)
    print(average_test_pred_list)

    # fig, ax = plt.subplots()

    gen_method_names = ["CustomDiffusion", "OMG+lora", "OMG+InstantID", "fastcomposer", "Mix-of-Show", "DreamBooth"]
    gen_method_names.reverse()
    linear_test_pred_list.reverse()
    average_test_pred_list.reverse()
    
    x = np.arange(len(gen_method_names))*1.4  # the label locations
    width = 0.50  # the width of the bars

    multiplier = 2
    fontsize = 18

    fig, ax = plt.subplots(figsize=(9, 9))

    offset = width * multiplier
    rects = ax.barh(x + offset, linear_test_pred_list, width, label="Ours(linear)", color='lightblue')
    ax.bar_label(rects, padding=3, fontsize=fontsize)
    multiplier += 1

    offset = width * multiplier
    rects = ax.barh(x + offset, average_test_pred_list, width, label="Ours(average)",color='steelblue')
    ax.bar_label(rects, padding=3, fontsize=fontsize)
    multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Total Score', fontsize=fontsize)
    ax.set_yticks(x+1+0.2, gen_method_names, fontsize=fontsize)
    ax.legend(loc='upper right', ncols=1, fontsize=fontsize)
    ax.set_ylim(0, 10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.tick_params(axis='x', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(reg_dir, "comparison.pdf"))
    plt.close()

def rank_plot(rank_list, reg_dir, eval_metric_list):
    # print(json_data)
    gen_method_names = ["CustomDiffusion", "OMG+lora", "OMG+InstantID", "fastcomposer", "Mix-of-Show", "DreamBooth"]
    eval_metric_names = eval_metric_list + ["Ours"] + ["UserPreference"]

    gen_method_names.reverse()
    eval_metric_names.reverse()
    rank_list.reverse()

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams.update({'font.size': 18})

    
    x = np.arange(len(gen_method_names))*2  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 2
    fontsize = 18

    fig, ax = plt.subplots(figsize=(9,12))
    colors = ['red', 'blue', 'pink', 'khaki', 'lightgray', 'mediumaquamarine', 'powderblue', 'thistle']

    for i, (method_name, rank) in enumerate(zip(eval_metric_names, rank_list)):
        print(method_name)
        rank.reverse()

        offset = width * multiplier
        print(x+offset)
        rects = ax.barh(
            x + offset, 
            rank, 
            width, 
            label=method_name, 
            color=colors[i])
        # ax.bar_label(rects, padding=3, fontsize=fontsize)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('rank', fontsize=fontsize)
    ax.set_yticks(x+1.2, gen_method_names, fontsize=fontsize)
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncols=2, fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.tick_params(axis='x', labelsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(os.path.join(reg_dir, "benchmark_rank.pdf"))
    plt.close()


def radar_factory(num_vars, frame='polygon'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines() # type: ignore
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def radar_chart(mean_list, reg_dir):
        
    gen_labels = ["CustomDiffusion", "OMG+lora", "OMG+InstantID", "fastcomposer", "Mix-of-Show", "DreamBooth"]

    colors = ['silver', 'skyblue', 'lightpink', 'orange', 'palegreen', 'yellow']

    ea_labels = [evaluation_aspects[f"score{i}"] for i in range(1,19)]
    N = len(ea_labels)
    theta = radar_factory(N, frame='polygon')
    print("theta:", theta)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(
        figsize=(9, 9), 
        subplot_kw=dict(projection='radar')
    )

    for d, color in zip(mean_list, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(ea_labels) # type: ignore
    ax.set_rgrids([1,2,3,4,5]) # type: ignore
    # add legend relative to top-left plot
    
    legend = ax.legend(gen_labels, loc='lower center', bbox_to_anchor=(0.5, 1.05),labelspacing=0.1, ncols=3)
    plt.tight_layout()
    plt.savefig(os.path.join(reg_dir, f"benchmark_raderchart.pdf"), bbox_inches="tight")
    plt.close()
