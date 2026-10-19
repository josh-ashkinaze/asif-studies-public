"""
Author: Joshua Ashkinaze

Description: A set of generic helper functions I use in projects.

Sections:
- Data viz
- Stats
- Input/output
- Pandas utilities
"""

import json
import logging
import re
import statistics as st
from collections import Counter
from datetime import datetime
import math
import textwrap
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap


# DATA VIZ
###################################
###################################
def make_aesthetic(hex_palette=None,
                   bold_title=False,
                   save_transparent=True,
                   font_scale=1,
                   font_hierarchy_ratio=1.2,
                   smallest_font_size=15,
                   text_color='#424242',
                   font_family='Arial',
                   figsize=None,
                   base_reference_size=(9, 6),
                   with_gridlines=False,
                   with_ticks=False,
                   ):
    """
    Make Seaborn look clean and auto-scale fonts based on figure size.

    Usage:
        >>> # Normal usage (defaults)
        >>> make_aesthetic()
        >>> # just plot matplot or seaborn graph here and things are nice!

        >>> # Auto-scaling usage:
        >>> # This will auto-calculate the font_scale to make the text look
        >>> # proportional to a 20x15 inch plot.
        >>> make_aesthetic(figsize=(20, 15))

        >>> # With gridlines and ticks enabled. Science submission guidelines sais they require ticks.
        >>> make_aesthetic(with_gridlines=True, with_ticks=True)

    Args:
        hex_palette (list or str, default=None): Color palette. Can be:
            - None: uses the 'default' palette
            - str: name of a built-in palette (e.g. 'wong', 'tableau10'). Pass any string
              to see available options.
            - list: custom list of hex color strings.
        bold_title (bool, default=False): Whether to make the title bold.
        save_transparent (bool, default=True): Save figures with transparent background.
        font_scale (float, default=1): Base scaling factor. Increase this to make all fonts bigger.
        font_hierarchy_ratio (float, default=1.2): Ratio to scale small/med/large fonts. There are three
            font sizes and this controls the contrast between them.
        smallest_font_size (int, default=15): The smallest font size (at base reference size).
        text_color (str, default='#424242'): Hex color for text.
        font_family (str, default='Arial'): Font family (falls back gracefully if unavailable).
        figsize (tuple, default=None): Optional (width, height) in inches. If provided, fonts
                         will auto-scale to maintain proportion with this size.
        base_reference_size (tuple, default=(9, 6)): The (width, height) used as the "standard"
                                     size for the smallest_font_size.
        with_gridlines (bool, default=False): Whether to show gridlines. When True,
                                gridlines are shown in the theme style (subtle, dashed).
        with_ticks (bool, default=False): Whether to show axis tick marks. When True,
                            ticks are shown in the theme style (outward, light gray).

    Palette Options:
        - 'default': custom 15-color palette
        - 'early_apple': vintage Apple logo colors (6)
        - 'tol': Paul Tol's colorblind-friendly scheme (9)
        - 'wong': Bang Wong 2011, Nature Methods — colorblind-safe (8)
        - 'ibm': IBM's colorblind-accessible palette (5)
        - 'tableau10': Tableau's default palette (10)
        - 'set2': ColorBrewer Set2 — soft/pastel (8)
        - 'dark2': ColorBrewer Dark2 — darker, higher contrast (8)

    Typolography Notes:

        Here is how the font sizes are determined:
        - Small font size: `smallest_font_size * font_scale`
        - Medium font size: `smallest_font_size * font_hierarchy_ratio * font_scale`
        - Large font size: `smallest_font_size * font_hierarchy_ratio^2 * font_scale`

        Based on trial and error, I set the title to be the average of medium and large to give it prominence without being
        overpowering.

        So:
        - font_scale scales all font sizes
        - font_hierarchy_ratio determines the relative sizes of small, medium, and large fonts.
        - smallest_font_size sets the baseline size for the smallest font at the base reference figure size. If you increase this, all fonts will be larger, but the relative hierarchy will be maintained.

        This site lists some good ratios:
        https://cieden.com/book/sub-atomic/typography/establishing-a-type-scale
    """

    # Reset to a clean baseline before applying custom settings.
    # sns.set() alone doesn't undo all prior rcParams changes.
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(style='ticks' if with_ticks else 'white', context='paper')

    ########## Font Resolution ##########
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if font_family not in available_fonts:
        font_family = plt.rcParamsDefault['font.sans-serif'][0]

    ########## Auto-Scaling Logic ##########
    if figsize is not None:
        ref_diag = math.hypot(base_reference_size[0], base_reference_size[1])
        target_diag = math.hypot(figsize[0], figsize[1])
        scale_factor = target_diag / ref_diag
        font_scale = font_scale * scale_factor
        plt.rcParams['figure.figsize'] = figsize

    ########## Color Palette ##########
    if hex_palette is None:
        hex_palette = PALETTES['default']
    elif isinstance(hex_palette, str):
        if hex_palette not in PALETTES:
            raise ValueError(f"Unknown palette '{hex_palette}'. Available: {list(PALETTES.keys())}")
        hex_palette = PALETTES[hex_palette]

    sns.set_palette(sns.color_palette(hex_palette))

    ########## Font Size Hierarchy ##########
    small = smallest_font_size * font_scale
    medium = small * font_hierarchy_ratio
    large = medium * font_hierarchy_ratio

    # Title sits between medium and large, prominent but not overpowering
    title = (medium + large) / 2

    plt.rcParams.update({
        # font settings
        'font.family': font_family,
        'font.weight': 'regular',
        'axes.labelsize': medium,
        'axes.titlesize': title,
        'xtick.labelsize': small,
        'ytick.labelsize': small,
        'legend.fontsize': small,
        'legend.title_fontsize': medium,
        'axes.titlecolor': text_color,
        'text.color': text_color,
        'xtick.labelcolor': text_color,
        'ytick.labelcolor': text_color,
        'axes.labelcolor': text_color,

        # spines/grids
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#BDBDBD',
        'axes.linewidth': 0.8 * font_scale,
        'axes.grid': with_gridlines,
        'grid.color': '#e8e8e8',
        'grid.alpha': 1.0,
        'grid.linestyle': '--',
        'grid.linewidth': 0.9 * font_scale,

        # ticks — outward to avoid clashing with data
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 4 * font_scale if with_ticks else 0,
        'ytick.major.size': 4 * font_scale if with_ticks else 0,
        'xtick.major.pad': 6 * font_scale,
        'ytick.major.pad': 6 * font_scale,
        'xtick.color': '#BDBDBD',
        'ytick.color': '#BDBDBD',

        # title
        'axes.titlelocation': 'left',
        'axes.titleweight': 'bold' if bold_title else 'regular',
        'axes.titlepad': 15 * font_scale,

        # fig — use constrained_layout only (autolayout conflicts)
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.2,
        'figure.constrained_layout.w_pad': 0.2,
        'figure.autolayout': False,

        # legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#E0E0E0',
        'legend.borderpad': 0.4,
        'legend.borderaxespad': 1.0,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.7,
        'legend.handletextpad': 0.5,

        # export
        'savefig.dpi': 300,
        'savefig.transparent': save_transparent,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,

        # mathtext — use resolved font
        'mathtext.fontset': 'custom',
        'mathtext.rm': font_family,
        'mathtext.it': f'{font_family}:italic',
        'mathtext.bf': f'{font_family}:bold',
    })

    return hex_palette


def smart_legend(ax=None, position='auto', **legend_kwargs):
    """Intelligent legend positioning that prevents overlap"""
    if ax is None:
        ax = plt.gca()

    legend_defaults = {
        'frameon': True,
        'framealpha': 0.95,
        'facecolor': 'white',
        'edgecolor': '#CCCCCC',
    }
    legend_defaults.update(legend_kwargs)

    if position == 'outside_right':
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', **legend_defaults)
    elif position == 'outside_bottom':
        legend = ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center',
                           ncol=legend_kwargs.get('ncol', 3), **legend_defaults)
    else:
        legend = ax.legend(loc='best', **legend_defaults)

    return legend


def clean_vars(s, how='title'):
    """
    Simple function to clean titles

    Params
    - s: The string to clean
    - how (default='title'): How to return string. Can be either ['title', 'lowercase', 'uppercase']

    Returns
    - cleaned string
    """
    assert how in ['title', 'lowercase', 'uppercase'], "Bad option!! see docs"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    s = s.replace('_', ' ')
    if how == 'title':
        return s.title()
    elif how == 'lower':
        return s.lower()
    elif how == 'upper':
        return s.upper()


def make_bold(x):
    """
    This function is used to make part of the title bold like as a subtitle.
    Basically, it's using latex to render a (sub)string bold in matplotlib.
    But make_aesthetic() should handle using Arial for math font, so it won't look weird.

    >>> full_title = f"{make_bold('Regression Coefficients of Estimated Prevalence From Multiverse')}\\n(Baseline is raw data with no weighting and no dropping)"
    """
    words = x.split()
    words = r'\ '.join([w for w in words])  # Escape backslash properly
    bold_str = f"$\\bf{{{words}}}$"  # Correctly format the f-string
    return bold_str


PALETTES = {
    # Vintage apple
    # https://www.schemecolor.com/apple-old-logo.php
    "early_apple": [
        "#009DDC",  # Blue Bell
        "#61BB46",  # Bright Fern
        "#FDB827",  # Amber Flame
        "#F5821F",  # Vivid Tangerine
        "#E03A3E",  # Scarlet Rush
        "#963D97",  # Grape Soda
    ],
    # My default
    "default": [
        "#00A896",  # Green
        "#826AED",  # Purple
        "#E3B505",  # Yellow
        "#89DAFF",  # Cyan
        "#F45B69",  # Red
        "#F18805",  # Orange
        "#D41876",  # Magenta
        "#020887",  # Blue
        "#7DCD85",  # Emerald
        "#E87461",  # Medium-bright orange
        "#7E6551",  # Coyote
        "#342E37",  # Dark grayish-purple
        "#F7B2AD",  # Melon
        "#D4B2D8",  # Pink lavender
        "#2C3531",  # Dark charcoal gray
    ],
    # Paul Tol palette
    # Corresponds to "Paul Tol Muted" here: https://colorteller.kausalflow.com/colors/
    "tol": ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
            "#882255", "#44AA99", "#999933", "#AA4499"],

    # Bang Wong (2011), Nature Methods — gold standard colorblind-safe
    # https://www.nature.com/articles/nmeth.1618
    "wong": [
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#009E73",  # Bluish Green
        "#F0E442",  # Yellow
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#CC79A7",  # Reddish Purple
        "#000000",  # Black
    ],
    # IBM's colorblind-accessible 5-color palette
    # https://lospec.com/palette-list/ibm-color-blind-safe
    "ibm": [
        "#648FFF",  # Blue
        "#785EF0",  # Purple
        "#DC267F",  # Magenta
        "#FE6100",  # Orange
        "#FFB000",  # Gold
        "#000000",  # Black

    ],
    # Tableau's default (10)
    # https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
    "tableau10": [
        "#4E79A7",  # Steel Blue
        "#F28E2B",  # Tangerine
        "#E15759",  # Red
        "#76B7B2",  # Teal
        "#59A14F",  # Green
        "#EDC948",  # Yellow
        "#B07AA1",  # Mauve
        "#FF9DA7",  # Pink
        "#9C755F",  # Brown
        "#BAB0AC",  # Gray
    ],
    # ColorBrewer Set2
    # https://emilhvitfeldt.github.io/r-color-palettes/discrete/RColorBrewer/Set2/
    "set2": [
        "#66C2A5",  # Mint
        "#FC8D62",  # Salmon
        "#8DA0CB",  # Periwinkle
        "#E78AC3",  # Pink
        "#A6D854",  # Yellow-Green
        "#FFD92F",  # Yellow
        "#E5C494",  # Tan
        "#B3B3B3",  # Gray
    ],
    # ColorBrewer Dark2
    # https://emilhvitfeldt.github.io/r-color-palettes/discrete/RColorBrewer/Dark2/index.html
    "dark2": [
        "#1B9E77",  # Teal
        "#D95F02",  # Orange
        "#7570B3",  # Purple
        "#E7298A",  # Pink
        "#66A61E",  # Green
        "#E6AB02",  # Amber
        "#A6761D",  # Brown
        "#666666",  # Gray
    ],
}

def clean_vars(s, how='title'):
    """
    Simple function to clean titles

    Params
    - s: The string to clean
    - how (default='title'): How to return string. Can be either ['title', 'lowercase', 'uppercase']

    Returns
    - cleaned string
    """
    assert how in ['title', 'lowercase', 'uppercase'], "Bad option!! see docs"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    s = s.replace('_', ' ')
    if how == 'title':
        return s.title()
    elif how=='lower':
        return s.lower()
    elif how=='upper':
    	return s.upper()

def make_bold(x):
    """
    This function is used to make part of the title bold like as a subtitle.
    Basically, it's using latex to render a (sub)string bold in matplotlib.
    But make_aesthetic() should handle using Arial for math font, so it won't look weird.
    
    >>> full_title = f"{make_bold('Regression Coefficients of Estimated Prevalence From Multiverse')}\\n(Baseline is raw data with no weighting and no dropping)"
    """
    words = x.split()
    words = r'\ '.join([w for w in words])  # Escape backslash properly
    bold_str = f"$\\bf{{{words}}}$"  # Correctly format the f-string
    return bold_str

###################################
###################################


# STATS
###################################
###################################

def cat_stats(data, include_n=True, digits=1, sort_by='frequency', reverse=True):
    """Calculate and format statistics for categorical data.

    This function analyzes categorical data by computing frequencies and percentages
    for each unique category. It returns both a formatted string for display and
    a dict. You can choose how you want to order the categories. Its the cat version of arraystats.

    Args:
        data (array-like): The input categorical data array. Can be a list,
            numpy array, or any iterable containing categorical values.
        include_n (bool, optional): Whether to include raw counts (n=X) in the
            formatted output string. Defaults to True.
        digits (int, optional): Number of decimal places for percentage rounding
            in the output. Defaults to 1.
        sort_by (str, optional): How to sort the results. Options are:
            - 'frequency': Sort by frequency/count (default)
            - 'alphabetical': Sort alphabetically by category name
            - 'original': Maintain order of first appearance in data
        reverse (bool, optional): Whether to reverse the sort order. Defaults to
            True for descending frequency (most common first).

    Returns:
        tuple: A tuple containing two elements:
            - str: Formatted string showing categories with percentages and
              optionally counts in the format "Category1 (45.2%; n=10), 
            - dict: Dictionary with categories as keys and dictionaries as values
              containing:
              - 'count': Raw frequency count
              - 'percentage': Exact percentage (float)  
              - 'percentage_rounded': Rounded percentage for display

    Usage:
        >>> data = ["Josh", "Josh", "Josh", "Sam"]
        >>> cat_stats(data)
        ('Josh (75.0%; n=3), Sam (25.0%; n=1)',
        {'Josh': {'count': 3, 'percentage': 75.0, 'percentage_rounded': 75.0}, 'Sam': {'count': 1, 'percentage': 25.0, 'percentage_rounded': 25.0}})
    """
    # Convert to numpy array for consistency
    data = np.array(data)
    # Count frequencies
    counter = Counter(data)
    total_count = len(data)

    # Calculate percentages and build results dictionary
    result_dict = {}
    for category, count in counter.items():
        percentage = (count / total_count) * 100
        result_dict[category] = {
            'count': count,
            'percentage': percentage,
            'percentage_rounded': round(percentage, digits)
        }

    # Sort results according to preference
    if sort_by == 'frequency':
        sorted_items = sorted(result_dict.items(),
                              key=lambda x: x[1]['count'],
                              reverse=reverse)
    elif sort_by == 'alphabetical':
        sorted_items = sorted(result_dict.items(),
                              key=lambda x: str(x[0]),
                              reverse=reverse)
    elif sort_by == 'original':
        # Maintain order of first appearance in the data
        seen = set()
        original_order = []
        for item in data:
            if item not in seen:
                seen.add(item)
                original_order.append(item)
        sorted_items = [(item, result_dict[item]) for item in original_order]
    else:
        raise ValueError("sort_by must be 'frequency', 'alphabetical', or 'original'")

    # Format output string
    result_parts = []
    for category, stats in sorted_items:
        if include_n:
            part = f"{category} ({stats['percentage_rounded']:.{digits}f}%; n={stats['count']})"
        else:
            part = f"{category} ({stats['percentage_rounded']:.{digits}f}%)"
        result_parts.append(part)

    result_string = ", ".join(result_parts)

    # Return both the formatted string and the full dictionary for programmatic use
    return result_string, {k: v for k, v in sorted_items}

def cat_stats_table(columns, data, **kwargs):
    """Build a summary table from multiple cat_stats calls.

    Args:
        columns (list): Column names to summarize.
        data (pd.DataFrame): DataFrame containing the columns.
        **kwargs: Passed through to cat_stats (digits, sort_by, include_n, etc).

    Returns:
        pd.DataFrame: Table with columns [column, category, count, percentage].

    Usage:
        >>> cat_stats_table(["treat_status", "is_mobile"], df)
    """
    rows = []

    for col in columns:
        _, stats = cat_stats(data[col].dropna(), **kwargs)
        for category, s in stats.items():
            rows.append({
                "column":     col,
                "category":   category,
                "count":      s["count"],
                "percentage": s["percentage_rounded"],
            })

    return pd.DataFrame(rows)

def age_bands(x):
    """Return an age-band label for a given age value.

    Bands (non-overlapping):
      - '18-34'  : ages 18 through 34 inclusive
      - '35-54'  : ages 35 through 54 inclusive
      - '55+'    : ages 55 and above

    Returns np.nan for missing, non-numeric or out-of-range (<18) ages.
    """
    # Guard against missing values
    if pd.isna(x):
        return np.nan

    try:
        age = float(x)
    except Exception:
        return np.nan

    if age < 18:
        return np.nan

    if 18 <= age <= 34:
        return "18-34"
    if 35 <= age <= 54:
        return "35-54"
    if age >= 55:
        return "55+"

    # Fallback (shouldn't be reached)
    return np.nan

def array_stats(data, digits=2, include_ci=False):
    """Calculate and print summary statistics for an array.

    This function computes basic descriptive statistics (mean, median, standard
    deviation, and mode) for a given array of data. It also provides an option
    to calculate and include a 95% confidence interval for the mean using
    bootstrap resampling.

    Args:
        data (array-like): The input data array for which statistics will be
            calculated. Can be a list, numpy array, or any array-like structure.
        digits (int, optional): Number of decimal places for rounding the
            calculated statistics. Defaults to 2.
        include_ci (bool, optional): Whether to include a 95% confidence
            interval for the mean using bootstrap resampling. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics with the
            following keys:
            - 'mean': The arithmetic mean of the data
            - 'median': The median (middle value) of the data  
            - 'sd': The sample standard deviation (using ddof=1)
            - 'mode': The most frequently occurring value
            - 'ci': (optional) A tuple containing the lower and upper bounds
              of the 95% confidence interval, only present if include_ci=True
    Prints:

        The function prints formatted statistics to the console like this--
        "M = 2.83, SD = 1.47, Mdn = 2.50" \n "Mode = 2.00" in addition to
        returning them as a dictionary. If include_ci=True, it also prints the
        confidence interval as "95% CI [1.83, 3.83]".

    Notes:
        When multiple modes exist, the function will print a warning message and use the first occurrence.
        The bootstrap uses Scipy 10K iterations and is bootstrapping the mean.

    Usage:
        >>> import numpy as np
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> stats = array_stats(data)
    """
    data = np.array(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    sd_val = np.std(data, ddof=1)

    # Calculate mode - handling cases with multiple modes
    try:
        mode_val = st.mode(data)
    except st.StatisticsError:
        # If multiple modes, use scipy's mode which returns the first occurrence
        print("Multiple modes found, using the first one.")
        mode_val = stats.mode(data, keepdims=True)[0][0]

    result = {
        'mean': round(mean_val, digits),
        'median': round(median_val, digits),
        'sd': round(sd_val, digits),
        'mode': round(mode_val, digits)
    }

    # Add confidence interval if requested
    if include_ci:
        def mean_func(x, axis):
            return np.mean(x, axis=axis)

        data_reshaped = np.array(data).reshape(-1, 1)
        bootstrap_result = bootstrap((data_reshaped,), mean_func,
                                     confidence_level=0.95,
                                     random_state=42,
                                     n_resamples=10 * 1000)
        ci_lower, ci_upper = bootstrap_result.confidence_interval
        result['ci'] = (round(float(ci_lower), digits), round(float(ci_upper), digits))

    print(f"M = {result['mean']:.{digits}f}, SD = {result['sd']:.{digits}f}, Mdn = {result['median']:.{digits}f}")
    print(f"Mode = {result['mode']:.{digits}f}")
    if include_ci and 'ci' in result:
        print(f"95% CI [{result['ci'][0]:.{digits}f}, {result['ci'][1]:.{digits}f}]")

    return result


def format_p_value(p, threshold=0.001, exact=True, digits=2):
    """
    Format p-values consistently for publication and presentation.
    
    This function standardizes p-value formatting following APA guidelines,
    with options for exact values or thresholding for very small p-values. The
    threshold is 0.001.
    
    Args:
        p (float): The p-value to format
        threshold (float, optional): Threshold below which to report as "p < threshold".
            Defaults to 0.001.
        exact (bool, optional): If True, always report exact p-value regardless of
            threshold. Defaults to True.
        digits (int, optional): Number of decimal places for exact p-values.
            Defaults to 3.
    
    Returns:
        str: Formatted p-value string (e.g., "p < .001", "p = .043", "p = .200")
    
    """
    # Handle edge cases
    if p < 0 or p > 1:
        raise ValueError("p-value must be between 0 and 1")

    # If exact=True, use scientific notation for values below threshold
    if exact:
        if p < threshold:
            sci_digits = 0 if p == float(f"{p:.0e}") else 1
            return f"p = {p:.{sci_digits}e}"
        else:
            return f"p = {p:.{digits}f}".replace("0.", ".")

    # If not exact and p below threshold, report as "p < threshold"
    else:
        if p < threshold:
            return f"p < {threshold}".replace("0.", ".")

    # But for values at or above threshold, report exact value
    formatted_p = f"{p:.{digits}f}".replace("0.", ".")
    return f"p = {formatted_p}"


def extreme_by_col(df, col, n_per_extreme=10):
   """Get top and bottom N rows sorted by column"""
   sorted_df = df.sort_values(col, ascending=False)
   
   top_n = sorted_df.head(n_per_extreme).copy()
   top_n['extreme'] = 'top'
   
   bottom_n = sorted_df.tail(n_per_extreme).copy()
   bottom_n['extreme'] = 'bottom'
   
   return pd.concat([top_n, bottom_n], ignore_index=True)

def extract_first_number(s):
    """Extract the first number from a string, return as float"""

    # check if already just a number
    if isinstance(s, (int, float)):
        return float(s)
    match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', s)
    if match:
        return float(match.group())
    else:
        return None


def format_p_value_stars(p, thresholds=(0.001, 0.01, 0.05)):
    """
    Format p-values as stars for significance levels.

    Args:
        p (float): The p-value to format.
        thresholds (tuple, optional): Thresholds for (***, **, *).
            Defaults to (0.001, 0.01, 0.05).

    Returns:
        str: Stars representing significance level ("***", "**", "*", or "")
    """
    if not (0 <= p <= 1):
        raise ValueError("p-value must be between 0 and 1")

    if p < thresholds[0]:
        return "***"
    elif p < thresholds[1]:
        return "**"
    elif p < thresholds[2]:
        return "*"
    else:
        return ""


def add_stars_to_df(df, p_cols, digits=3):
    """Return a copy of df with significance stars appended to each p-value column.

    For each column name in p_cols, the numeric p-value is replaced with a
    string formatted as ``"0.023*"`` (or ``"<0.001***"`` for very small values).
    Non-p columns are left untouched.

    Args:
        df (pd.DataFrame): Table containing at least the columns in p_cols.
        p_cols (list[str]): Column names whose values are p-values to annotate.
        digits (int): Decimal places for the formatted p-value. Default 3.

    Returns:
        pd.DataFrame: Copy with p_cols replaced by star-annotated strings.

    Example:
        >>> ttest_df = add_stars_to_df(ttest_df, ["AIM vs. Vignette p", "AIM vs. Placebo p"])
    """
    out = df.copy()
    for col in p_cols:
        out[col] = out[col].apply(
            lambda p: f"{p:.{digits}f}{format_p_value_stars(p)}"
        )
    return out


def reverse_scale(value, original_min, original_max):
    """Reverse scale a value from original range to new range"""
    return original_max - (value - original_min)

def rescale(value, original_min, original_max, new_min=0, new_max=1):
    """Rescale a value from original range to new range"""
    scaled_value = (value - original_min) / (original_max - original_min)
    return new_min + (scaled_value * (new_max - new_min))

def icw_index(arrays, reference_mask=None):
    """
    Create ICW index from multiple indicator arrays.

    Follows Schwab (2020) and Stata's swindex with restrictions:
    - No missing values allowed in input arrays

    Args:
    - arrays: List of 1D arrays (indicators, same length, no NaNs)
    - reference_mask: Boolean array indicating reference group for covariance calculation. This is equivalent to STATA's swindex
    normby() option. If None, uses all observations as the reference group.

    Returns:
    - 1D array (ICW index) normalized to mean=0, sd=1 for reference group.
        - If reference is full sample, entire index has mean=0, sd=1.
        - If reference is (eg) control group, then control group has mean=0, sd=1 and treated group ICWs are relative to that.

    Usage:
        x1 = np.random.rand(100)
        x2 = np.random.rand(100)
        index = icw_index([x1, x2])

        # Example using Pandas dataframes and reference group normalization
        df = pd.DataFrame({'var1': np.random.rand(100),
                            'var2': np.random.rand(100),
                            'treat': np.random.randint(0, 2, size=100)})
        ref_mask = (df['treat'] == 0).values

        # Full sample normalization, no reference group. Entire index is distributed M=0, SD=1
        df['icw'] = icw_index([df['var1'].values, df['var2'].values])

        # User-specified reference group normalization. Control group is distributed M=0, SD=1 and treatment group is relative to that.
        df['icw_control_reference'] = icw_index([df['var1'].values, df['var2'].values],
                                         reference_mask=ref_mask)
    """
    import numpy as np

    # # Validate arrays
    # for arr in arrays:
    #     if np.isnan(arr).any():
    #         raise ValueError("Input arrays must not contain NaN values.")

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")

    # Validate and set reference mask
    if reference_mask is not None:
        if len(reference_mask) != lengths[0]:
            raise ValueError("Reference mask must be same length as input arrays.")
        if reference_mask.dtype != bool:
            raise ValueError("Reference mask must be a boolean array.")
        if np.isnan(reference_mask).any():
            raise ValueError("Reference mask must not contain NaN values.")
    else:
        reference_mask = np.ones(lengths[0], dtype=bool)

    # Stack indicators and extract reference subset
    indicators = np.column_stack(arrays)
    # replace missing with 0:
    indicators = np.nan_to_num(indicators, nan=0.0)
    ref_indicators = indicators[reference_mask, :]

    # Normalize indicators using reference group stats
    means = np.mean(ref_indicators, axis=0)
    sds = np.std(ref_indicators, axis=0, ddof=1)
    normalized_all = (indicators - means) / sds

    # Compute weights from inverse covariance of reference group
    cov_matrix = np.cov(normalized_all, rowvar=False)
    # Use pseudo-inverse for singular matrices
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)
    weights = np.sum(inv_cov, axis=1)

    # Construct index for all observations
    index_all = np.dot(normalized_all, weights)

    # Normalize using reference group statistics
    index_ref = index_all[reference_mask]
    index_mean = np.mean(index_ref)
    index_sd = np.std(index_ref, ddof=1)
    final_index = (index_all - index_mean) / index_sd

    return final_index




###################################
###################################

# I/O UTILS
###################################
###################################
def list2text(data_list, filename):
    """Write list to text file, one item per line"""
    with open(filename, 'w') as f:
        f.write('\n'.join(map(str, data_list)))

def text2list(filename):
    """Read text file into list, one line per item"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]
	    
def timestamp(style='readable'):
    """Return current timestamp string for filenames"""
    if style == 'readable':
        return datetime.now().strftime('%Y-%m-%d__%H.%M.%S')
    elif style == 'compact':
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    elif style == 'date_only':
        return datetime.now().strftime('%Y-%m-%d')

def dict2json(data_dict, filename):
    """Save dictionary to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=2)

def json2dict(filename):
    """Load dictionary from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def log_and_print(message):
    """Log a message using the current logger and print it to console."""
    logger = logging.getLogger()
    logger.info(message)
    print(message)
###################################
###################################
