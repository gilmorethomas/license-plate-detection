from src.logger import logger as logging 
from os import path, makedirs
import plotly.graph_objects as pgo
import plotly.express as px
from src.globals import LICENSE_PLATE_GLOBALS as LPG
import numpy as np
import pandas as pd
def _make_title_replacements(title):
    # Replace 'newline' with a newline character
    title = title.replace('newline' , '<br>')
    # Replace underscores with space 
    title = title.replace('_', ' ')
    # Capitalize the first letter of each word
    title = title.title()
    # Replace 'Vs' with 'vs'
    title = title.replace('Vs', 'vs')
    return title

def pie_chart(data, values, names, title, filename, output_dir, save_png=True, save_html=True):
    fig = px.pie(data, values=values, names=names, title=title)
    finalize_plot(fig, title, filename, output_dir, save_png=save_png, save_html=save_html)

def histogram(data, x, title, filename, output_dir, save_png=True, save_html=True, d_levels=None, num_bins=10):
    # Calculate our bins. This ideally is num_bins equally spaced bins. To deal with the fact htat d_levels may not be equally spaced, we will create a separate bin for each d_level
    # Replace None values with a placeholder
    data[x] = data[x].replace([None], 'None')
    # Replace nan values with a placeholder
    data[x] = data[x].replace([np.nan], 'nan')
    
    # Calculate bins for numeric values, which corresopnds to not nan and not None values
    numeric_data = data[data[x] != 'None'][x].astype(float)
    numeric_data = numeric_data[~np.isnan(numeric_data)]
    # If we have bins > number of unique values, we will use the number of unique values as the number of bins
    if num_bins > len(numeric_data.unique()):
        num_bins = len(numeric_data.unique())
    bins = np.linspace(numeric_data.min(), numeric_data.max() + (numeric_data.max() - numeric_data.min())/ num_bins, num_bins + 1)
    # Add in our d_level bins
    if d_levels is not None:
        # Concatenate the bins with the unique values of the d_levels
        bins = np.concatenate([d_levels, bins])
    # Get rid of duplicates, which may have been created by d levels
    bins = np.unique(bins)

    # Sort the bins
    bins = np.sort(bins)


    # Create the histogram data. Make this left inclusive and right inclusive. Use only numeric data
    hist_data = pd.cut(numeric_data, bins, right=False).sort_values().astype(str)
    
    # Add the 'None' bin
    hist_data = pd.concat([
        pd.Series(['None'] * data[data[x] == 'None'].shape[0]), 
        pd.Series(['nan'] * data[data[x] == 'nan'].shape[0]),
        hist_data])
    hist_data.name = x + '_hist'
    # Now that we have a series, merge it with the dataframe based on the index of the sorted data
    hist_data = pd.merge(hist_data, data, left_index=True, right_index=True)
    # hist_data_proc = hist_data.astype(str)
    # hist_data_proc = pd.Categorical(hist_data_proc, categories=categories, ordered=True)
    

    # Create the histogram using the series
    fig = px.histogram(hist_data, title=title, x=x + '_hist')
    finalize_plot(fig, title, filename, output_dir, save_png=save_png, save_html=save_html)


    # # If there are d_levels, we want to create separate bins for these levels
    # fig = pgo.Figure()
    # if d_levels is not None:
    #     for level in d_levels:
    #         fig.add_trace(pgo.Histogram(x=data[data['d_level'] == level][x], name=level))
    # # Add the rest of the bins. We want the number of other bins to be num_bins
    # fig.add_trace(pgo.Histogram(x=data[x], name='Other'))
    # finalize_plot(fig, title, filename, output_dir, save_png=save_png, save_html=save_html)
    # If there are special x callouts, use them
def bar_plot(data, x, y, title, filename, output_dir, save_png=True, save_html=True, xaxis_title=None, yaxis_title=None):
    fig = px.bar(data, x=x, y=y, title=title)
    finalize_plot(fig, title, filename, output_dir, save_png=save_png, save_html=save_html, xaxis_title=xaxis_title, yaxis_title=yaxis_title, plot_type='bar')

def finalize_plot(fig, title, filename, output_dir, save_png=True, save_html=True, make_title_replacements=True, plot_type=None, xaxis_title=None, yaxis_title=None):
    """Writes a plot to a png and html file

    Args:
        fig (plotly.graph_objects.Figure): The plotly figure to save
        filename (str, path-like): The filename to save the plot as
        output_dir (str, path-like): The directory to save the plot in
        save_png (bool, optional): Whether or not to save png. Defaults to True.
        save_html (bool, optional): Whether or not to save html. Defaults to True.

    Returns:
        _type_: _description_
    """     
    if fig is None: 
        logging.error("No figure provided")
    #logging.info(f"Finalizing plot {filename}")
    fig.update_layout(
        title=title,
        template='presentation',  # Use dark theme
        hovermode='closest',  # Show hover information for the closest point
    )
    if plot_type is not None and plot_type == 'bar':
        fig.update_layout(barmode='group')
        fig.update_xaxes(tickangle=45, automargin=True)

    if xaxis_title is not None:
        fig.update_xaxes(title=xaxis_title)
    if yaxis_title is not None:
        fig.update_yaxes(title=yaxis_title)
    html_dir = path.join(output_dir, 'html')
    png_dir = path.join(output_dir, 'png')
    if not path.exists(output_dir):
            makedirs(output_dir)
    if save_html and not path.exists(html_dir):
            makedirs(html_dir)
    if save_png and not path.exists(png_dir):
            makedirs(png_dir)
    if make_title_replacements: 
        title = _make_title_replacements(title)
    if save_png and LPG.PNG_WORKS:
        try:
            fig.write_image(path.join(png_dir, f'{filename}.png'))
        except Exception as e:
            logging.error(f"Failed to write image {filename}.png, bypassing. Error: {e}")
            LPG.PNG_WORKS = False
    if save_html: 
        fig.write_html(path.join(html_dir, f'{filename}.html'))