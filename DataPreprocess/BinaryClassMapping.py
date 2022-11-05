"""
    Module to map attack to either 0 or 1
    Input:
        pandas dataframe with 'attack' col
    output:
        pandas dataframe with new col 'attack_class' with 0 (normal) or 1 (attack)
    exception handling
        exception will be thrown when given dataset doesnt have 'attack' col
"""

def BinMap(df):
    df['attack_class']  = df['attack'].apply(lambda v: 0 if v == "Normal" else 1)
    return df