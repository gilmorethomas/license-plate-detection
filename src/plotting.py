from src.logger import logger as logging 
from os import path, makedirs
import plotly 
import plotly.express as px

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
    if save_png:
        fig.write_image(path.join(png_dir, f'{filename}.png'))
    if save_html: 
        fig.write_html(path.join(html_dir, f'{filename}.html'))