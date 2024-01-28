import pandas as pd
import glob
import os

def load_all_csv_in_dir(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files in a directory and concatenate them into a single DataFrame.

    Parameters:
    data_dir (str): The directory path where the CSV files are located.

    Returns:
    pandas.DataFrame: A DataFrame containing the concatenated data from all CSV files.
    """
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []

    for filename in all_files:
        date_str = os.path.basename(filename).split(".")[0]
        date = pd.to_datetime(date_str, format="%Y-%m-%d")
        df = pd.read_csv(filename, index_col=None, header=0)
        df["date"] = date
        df_list.append(df)

    return pd.concat(df_list, axis=0, ignore_index=True)

def load_csv_file(filename: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(filename, index_col=None, header=0)