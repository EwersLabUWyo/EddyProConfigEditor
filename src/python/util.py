"""utility functions for eddyproconfigeditor"""

from collections.abc import Sequence
import datetime

from pandas import DataFrame

def or_isinstance(object, *types):
    """helper function to chain together multiple isinstance arguments
    e.g. or_isinstance(a, float, int) does the same as (isinstance(a, float) or isinstance(a, int))"""
    for t in types:
        if isinstance(object, t):
            return True
    return False

def in_range(v, interval):
    """helper function to determine if a value is in some interval.
    intervals are specified using a string:
    (a, b) for an open interval
    [a, b] for a closed interval
    [a, b) for a right-open interval
    etc.

    use inf as one of the interval bounds to specify no bound in that direction, but I don't know why you wouldn't just use > at that point"""

    assert interval.strip()[0] in ['(', '['] and interval.strip()[-1] in [')', ']'], 'interval must be a string of the form (a, b), [a, b], (a, b], or [a, b)'
    # remove whitespace
    interval = interval.strip()

    # extract the boundary conditions
    lower_bc = interval[0]
    upper_bc = interval[-1]

    # extract the bounds
    interval = [float(bound.strip())
                for bound in interval[1:-1].strip().split(',')]
    lower, upper = interval
    if lower == float('inf'):
        lower *= -1

    bounds_satisfied = 0
    if lower_bc == '(':
        bounds_satisfied += (lower < v)
    else:
        bounds_satisfied += (lower <= v)

    if upper_bc == ')':
        bounds_satisfied += (v < upper)
    else:
        bounds_satisfied += (v <= upper)

    return bounds_satisfied == 2

def compare_configs(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """compare differences between two configs
    
    Parameters
    ----------
    df1, df2: pandas.DataFrame objects output by EddyproConfigEditor.to_pandas()

    Returns
    -------
    pandas.DataFrame containing lines that differed between df1 and df2
    """
    df1_new = df1.loc[df1['Value'] != df2['Value'],
                      ['Section', 'Option', 'Value']]
    df2_new = df2.loc[df1['Value'] != df2['Value'],
                      ['Section', 'Option', 'Value']]
    name1 = df1['Name'].values[0]
    name2 = df2['Name'].values[0]
    df_compare = (
        df1_new
        .merge(
            df2_new, 
            on=['Section', 'Option'], 
            suffixes=['_' + name1, '_' + name2])
        .sort_values(['Section', 'Option']))
    return df_compare

def compute_date_overlap(
        interval_1: Sequence[str, str] | Sequence[datetime.datetime, datetime.datetime], 
        interval_2: Sequence[str, str] | Sequence[datetime.datetime, datetime.datetime]) -> datetime.timedelta:
    """given two time intervals of strings or datetime objects, compute their overlap and report them as a timdelta object
    Strings must be of the form YYYY-mm-dd HH:MM"""

    # assure inputs conform to standards
    assert isinstance(interval_1, Sequence), 'intervals must be sequences of strings or datetimes'
    assert isinstance(interval_2, Sequence), 'intervals must be sequences of strings or datetimes'
    assert len(interval_1) == 2, 'intervals must be length 2'
    assert len(interval_2) == 2, 'intervals must be length 2'
    interval_1 = list(interval_1)
    interval_2 = list(interval_2)
    for i in range(2):
        assert or_isinstance(interval_1[i], str, datetime.datetime), 'inputs must be strings of format YYYY-mm-dd HH:MM or datetime.datetime objects'
        assert or_isinstance(interval_2[i], str, datetime.datetime), 'inputs must be strings of format YYYY-mm-dd HH:MM or datetime.datetime objects'
        if isinstance(interval_1[i], str):
            assert len(interval_1[i].strip()) == 16, 'inputs must be strings of format YYYY-mm-dd HH:MM'
        if isinstance(interval_2[i], str):
            assert len(interval_2[i].strip()) == 16, 'inputs must be strings of format YYYY-mm-dd HH:MM'
    
    # convert to datetime objects
    for i in range(2):
        if isinstance(interval_1[i], str):
            interval_1[i] = datetime.datetime.strptime(interval_1[i].strip(), r'%Y-%m-%d %H:%M')
        if isinstance(interval_2[i], str):
            interval_2[i] = datetime.datetime.strptime(interval_2[i].strip(), r'%Y-%m-%d %H:%M')

    # compute overlap
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    overlap = end - start

    return overlap

