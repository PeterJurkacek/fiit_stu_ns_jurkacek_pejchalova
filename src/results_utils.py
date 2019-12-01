import numpy as np
import pandas as pd
from pathlib import Path
import src.config as config
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
pd.set_option('max_colwidth', 15)

def experiment_hystory_table(experiment_name, test: bool):
    experiment_dir = config.project_path / 'logs' / experiment_name
    logging.debug(f"Checking experiment directory: {experiment_dir}, exist: {experiment_dir.exists()}")
    df = pd.concat([parse_metadata(csv_file_path, test) for csv_file_path in experiment_dir.glob("**/history.csv")])
    df.sort_values(by=['max_acc'], ascending=True, inplace=True)
    table_name = "Testovacej sade" if test else "Trénovacej sade"
    return f"Úspešnosť na {table_name}", df

def parse_metadata(csv_file_path: Path, test: bool):
    model_name = csv_file_path.parent.stem.split('_')[0]
    logging.debug(f"Loading model {model_name} history from: {csv_file_path}")
    df = pd.read_csv(csv_file_path, sep=';')

    #print(df.columns)
    val = 'val_' if test else ''
    if 'categorical_accuracy' in df.columns:
        column = val + 'categorical_accuracy'
    elif 'accuracy' in df.columns:
        column = val + 'accuracy'

    columns = ['epoch', column]
    df = df[columns]
    df['model_name'] = model_name
    return create_statistic(df, column)


def create_statistic(df, column):
    logging.info(f"Loading statistics for column: {column}")
    return df.groupby('model_name').apply(lambda x: pd.Series({
        "max_acc": x[column].max(),
        "average_acc": x[column].mean(),
        "best_epoch": find_best_epoch(x, column),
    }))


def find_best_epoch(df, column_name):
    max = df[column_name] == df[column_name].max()
    nieco = df[max].head(1)['epoch'].values[0]
    return nieco

def color_up(val, required_value):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    return 'background-color: yellow' if val > required_value else ''


def background_up(series):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = series == series.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# experiment_hystory_table('experiment_1_trainable_false')

def plot_results(df, title):
    print(df.tail(2))
    width = 0.35  # the width of the bars
    df = df.round(2)
    labels = df.index
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, df['average_acc'], width, label='Priemerná úspešnosť')
    rects2 = ax.bar(x + width / 2, df['max_acc'], width, label='Maximálna dosiahnutá úspešsnoť')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f"Úspešnosť jednotlivých modelov (Accuracy)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def plot_distributions(purpose):
    path = config.project_path / 'notebooks' / 'vyhodnotenie' / 'distribution.csv'
    df = pd.read_csv(path, index_col=['distribution'])
    df = df[purpose].groupby('distribution').sum()
    # Pie chart
    labels = df.index
    counts = df['count']
    # only "explode" the 2nd slice (i.e. 'Hogs')
    explode = [0] * len(counts)
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()


def fancy_plot_distributions(purpose):
    path = config.project_path / 'notebooks' / 'vyhodnotenie' / 'distribution.csv'
    df = pd.read_csv(path, index_col=['distribution'])
    mask = df['purpose'] == purpose
    df = df[mask]
    # Pie chart
    smth = df.groupby('distribution').sum()
    # print(f"smth: {smth}")
    labels = smth.index.values
    counts = smth['count'].values
    classes_counts = df['count'].values
    classes = df['class'].values
    # print(f"labels: {labels}")
    # print(f"counts: {counts}")
    # print(f"classes_counts: {classes_counts}")
    # print(f"classes: {classes}")
    colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff']
    colors_classes = ['#c2c2f0','#ffb3e6','#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']

    explode_distributions = [0.2] * len(labels)
    explode_classes = [0.1] * len(classes)
    # Plot
    plt.pie(counts, labels=labels, colors=colors, startangle=90, frame=True, explode=explode_distributions, radius=6)
    plt.pie(classes_counts, labels=classes, colors=colors_classes, startangle=90, explode=explode_classes, radius=3, autopct='%1.f%%')
    # Draw circle
    centre_circle = plt.Circle((0, 0), radius=1.5, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(11, 8)
    fig.gca().add_artist(centre_circle)

    plt.axis('equal')
    plt.title(purpose)
    plt.tight_layout()
    plt.show()
