import os
import ntpath
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from scipy.stats import binned_statistic_2d
from matplotlib import ticker as mticker
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

#--------------------------------------------------------------------------------------------------------------------------
def plot_param():
    # Set the line width of plotted lines to 1.1
    mpl.rcParams['lines.linewidth'] = 1.1

    # Set the direction of x-axis ticks to be outward
    mpl.rcParams['xtick.direction'] = 'out'

    # Set the direction of y-axis ticks to be outward
    mpl.rcParams['ytick.direction'] = 'out'

    # Disable ticks on the top side of the plot
    mpl.rcParams['xtick.top'] = False

    # Disable ticks on the right side of the plot
    mpl.rcParams['ytick.right'] = False

    # Set the size of the figure to [4, 3] inches
    mpl.rcParams['figure.figsize'] = [6.8, 5.1]

    # Set the DPI (dots per inch) of the figure to 200
    mpl.rcParams['figure.dpi'] = 200

    # Set the DPI (dots per inch) for saving figures to 300
    mpl.rcParams['savefig.dpi'] = 300

    # Set the font size to 11
    mpl.rcParams['font.size'] = 15

    # Set the font to Arial and use sans-serif as the font family
    mpl.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

    # Set the math text font set to 'stixsans'
    mpl.rcParams['mathtext.fontset'] = 'stixsans'

    return None

#--------------------------------------------------------------------------------------------------------------------------
""" The function set_axes(var_ax) is a function that customizes the axes of a plot represented by the var_ax variable. """

def set_axes(var_ax, xlog = False):
    # Set the color of the bottom spine (x-axis) to black
    var_ax.spines['bottom'].set_color('black')

    # Set the color of the top spine to black
    var_ax.spines['top'].set_color('black')

    # Set the color of the right spine to black
    var_ax.spines['right'].set_color('black')

    # Set the color of the left spine to black
    var_ax.spines['left'].set_color('black')

    # Set tick parameters for both major and minor ticks
    var_ax.tick_params(which='both')

    # Set the length, direction, and width of major ticks
    var_ax.tick_params(which='major', length=6.5, direction='out', width=1.1)

    # Set the length, direction, color, and width of minor ticks
    var_ax.tick_params(which='minor', direction='out', length=4.5, color='gray', width=1.0)

    # Set the minor locator for the x-axis
    var_ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Set the minor locator for the y-axis
    var_ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Set the font name to "Arial" for all x-axis tick labels
    for tick in var_ax.get_xticklabels():
        tick.set_fontname("Arial")

    # Set the font name to "Arial" for all y-axis tick labels
    for tick in var_ax.get_yticklabels():   
        tick.set_fontname("Arial")

    if xlog == True:
        var_ax.set_xscale('log')
        var_ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
        var_ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

    # Return the modified axes object
    return var_ax

#--------------------------------------------------------------------------------------------------------------------------

""" The function get_csv_files(pathname) is used to retrieve a list of CSV file paths within a specified directory or its subdirectories.  """

def get_csv_files(pathname):
    # Create an empty list to store file paths
    files = []

    # Walk through the directory and its subdirectories
    # r = root directory, d = directories, f = files
    for r, d, f in os.walk(pathname):
        # Iterate over each file in the current directory
        for file in f:
            # Check if the file has a .csv extension
            if '.csv' in file:
                # Append the full path of the CSV file to the list
                files.append(os.path.join(r, file))

    # Return the list of CSV file paths
    return files

#--------------------------------------------------------------------------------------------------------------------------
""" The function CalFitIndex(y_data, yfit) calculates several performance metrics
to evaluate the goodness of fit between the predicted values (yfit) and the
actual values (y_data).
It computes the mean absolute error (MAE), mean squared error (MSE),
coefficient of determination (R-squared), explained variance score (EVS),
and standard deviation of the residuals.
The computed metrics are rounded to four decimal places and returned as a list (row_r)
containing R-squared, MSE, MAE, and standard deviation. """


def CalFitIndex(y_data, yfit):
    mae = mean_absolute_error(y_data, yfit)

    # Compute the mean squared error (MSE)
    mse = mean_squared_error(y_data, yfit)

    # Compute the coefficient of determination (R-squared)
    r2 = r2_score(y_data, yfit)

    # Compute the explained variance score (EVS)
    evs = explained_variance_score(y_data, yfit)
        
    # standard deviation 
        
    std_dev = round(float(np.std(np.array(y_data) - np.array(yfit))), 4)

    # Round the computed metrics to the desired decimal places
    mae = round(float(mae), 4)
    mse = round(float(mse), 4)
    r2 = round(float(r2), 4)
    evs = round(float(evs), 4)

    row_r = []
    row_r.append(r2)
    row_r.append(mse)
    row_r.append(mae)
    row_r.append(std_dev)
    return row_r
#--------------------------------------------------------------------------------------------------------------------------

xcolors = ['#000000', '#1E90FF', '#F00B5F', '#FF4500', '#4C6BBF', '#1E7037', '#A9A9A9', '#DC143C', '#778899', '#A0522D', '#7C8F8F', '#DB7093', \
'#8B008B', '#6A5ACD', '#2E8B57', '#DC143C', '#43579A', '#8B439A', '#BDB76B','#5AAAAB']

def contour_color_map():
    # colors = ['white', '#1E90FF', '#F00B5F', '#FF4500', '#4C6BBF', '#1E7037', '#A9A9A9']
    colors = ['white', '#1E90FF', '#F00B5F', '#FF4500', '#4C6BBF', 'blue']
    cmap = mcolors.ListedColormap(colors, name='custom_colormap', N=7)
    return cmap