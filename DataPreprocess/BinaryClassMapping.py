"""
    Module to map attack to either 0 or 1
    Input:
        pandas dataframe with 'attack' col
    output:
        pandas dataframe with new col 'attack_class' with 0 (normal) or 1 (attack)
    exception handling
        exception will be thrown when given dataset doesnt have 'attack' col
"""

from pandas.core.frame import DataFrame

NSL_TARGET = "attack_class"
UNSW_TARGET = "attack_cat"
mapping = {'ipsweep': 'Probe','satan': 'Probe','nmap': 'Probe','portsweep': 'Probe','saint': 'Probe','mscan': 'Probe',
        'teardrop': 'DoS','pod': 'DoS','land': 'DoS','back': 'DoS','neptune': 'DoS','smurf': 'DoS','mailbomb': 'DoS',
        'udpstorm': 'DoS','apache2': 'DoS','processtable': 'DoS',
        'perl': 'U2R','loadmodule': 'U2R','rootkit': 'U2R','buffer_overflow': 'U2R','xterm': 'U2R','ps': 'U2R',
        'sqlattack': 'U2R','httptunnel': 'U2R',
        'ftp_write': 'R2L','phf': 'R2L','guess_passwd': 'R2L','warezmaster': 'R2L','warezclient': 'R2L','imap': 'R2L',
        'spy': 'R2L','multihop': 'R2L','named': 'R2L','snmpguess': 'R2L','worm': 'R2L','snmpgetattack': 'R2L',
        'xsnoop': 'R2L','xlock': 'R2L','sendmail': 'R2L',
        'normal': 'Normal'}
def BinMap(df, target):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        if target == NSL_TARGET:
            df[target]  = df[target].apply(lambda v: 0 if v == "normal" else 1)
        elif target == UNSW_TARGET:
            df[target]  = df[target].apply(lambda v: 0 if v == "Normal" else 1)
    except TypeError as ex:
        print(f"Expected type of arg: pandas.core.frame.DataFrame {str(ex)}")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df
def BinMapM(df, target):
# map label to either 0 or 1    
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        if target == NSL_TARGET:
            df[target] = df[target].apply(lambda v: mapping[v])
    except TypeError as ex:
        print(f"Expected type of arg: pandas.core.frame.DataFrame {str(ex)}")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df
def resultMap(df, target):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        df[target]  = df[target].apply(lambda v: "Normal" if v == 0 else "Attack")
    except TypeError as ex:
        print("Expected type of arg: pandas.core.frame.DataFrame")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df

def resultMapM(df, target):
    # map label to either 0 or 1
    try:
        if not isinstance(df, DataFrame):
            raise TypeError
        if target == NSL_TARGET:
            df[target]  = df[target].apply(lambda v: mapping[v])
    except TypeError as ex:
        print("Expected type of arg: pandas.core.frame.DataFrame")
    except KeyError:
        print("Please ensure df has col named 'label'")
    else:
        return df