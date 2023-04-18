# imports
import pandas as pd
import wrangle_wine as w

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def quality_dis(train):
    '''
    Gets barplot of wine qualities
    '''
    
    # establish colors
    colors = ['firebrick', 'peachpuff']
    
    # set pallette colors
    sns.set_palette(sns.color_palette(colors))
    
    # create a histogram
    sns.histplot(train, x='quality', hue='white_wine', multiple='stack',
                 bins=4, discrete=True)
    
    # set the legend
    plt.legend(labels = ['White Wines','Red Wines'])
    
    # set the title
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Wine Qualities')
    plt.ylabel('Wine Count')
    plt.show()

    
def volatile_acidity_vis(train):
    '''
    Gets boxplotqith quality and volatile acidity
    '''
    # set the target
    target = 'quality'
    
    # create a boxplot with train dataset
    sns.boxplot(data = train, x = train[target], y = train['volatile_acidity'], 
                palette='rocket')
    
    # set title and labels
    plt.title('Volatile Acidity vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Volatile Acidity')
    
    # show the plot
    plt.show()

def alcohol_vis(train):
    '''
    Get a box plot visualization with quality and alcohol content
    '''
    # set the target
    target = 'quality'

    # create a box plot with the train dataset
    sns.boxplot(data = train, x = train[target], y = train['alcohol'], 
                palette='rocket')
    
    # set the titles and labels
    plt.title('Alcohol vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Alcohol')
    
    # show the visual
    plt.show()

def density_vis(train):
    '''
    Get a box plot visual of wine density and wine quality
    '''
    # set target
    target = 'quality'

    # create boxplot with the train dataset
    sns.boxplot(data = train, x = train[target], y = train['density'], palette='rocket')
    
    # set the title and the lables
    plt.title('Density vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Density')
    
    # show the visualization
    plt.show()

def chlorides_vis(train):
    '''
    Creates a boxplot visualization using wine quality and chlorides
    '''
    # set the target
    target = 'quality'
    
    # create boxplot with the train dataset
    sns.boxplot(data = train, x = train[target], y = train['chlorides'], palette='rocket')
    
    # set the title and the lables 
    plt.title('Chlorides vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Chlorides')
    
    # show the visuals
    plt.show()
