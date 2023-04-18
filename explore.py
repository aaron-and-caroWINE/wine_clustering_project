import pandas as pd
import wrangle_wine as w

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def quality_dis(train):

    colors = ['firebrick', 'peachpuff']
    sns.set_palette(sns.color_palette(colors))
    sns.histplot(train, x='quality', hue='white_wine', multiple='stack', bins=4, discrete=True)
    plt.legend(labels = ['White Wines','Red Wines'])
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Wine Qualities')
    plt.ylabel('Wine Count')
    plt.show()

    
def volatile_acidity_vis(train):
    target = 'quality'

    sns.boxplot(data = train, x = train[target], y = train['volatile_acidity'], palette='rocket')
    plt.title('Volatile Acidity vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Volatile Acidity')
    plt.show()

def alcohol_vis(train):
    target = 'quality'

    sns.boxplot(data = train, x = train[target], y = train['alcohol'], palette='rocket')
    plt.title('Alcohol vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Alcohol')
    plt.show()

def density_vis(train):
    target = 'quality'

    sns.boxplot(data = train, x = train[target], y = train['density'], palette='rocket')
    plt.title('Density vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Density')
    plt.show()

def chlorides_vis(train):
    target = 'quality'

    sns.boxplot(data = train, x = train[target], y = train['chlorides'], palette='rocket')
    plt.title('Chlorides vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Chlorides')
    plt.show()
