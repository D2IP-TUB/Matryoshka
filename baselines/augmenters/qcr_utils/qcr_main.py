from pathlib import Path
import pandas as pd
from typing import List, Optional
import psycopg as pg
import time


def search_qcr(
    query_df: pd.DataFrame,
    cursor: pg.Cursor,
    result_size: int,
    query_sketch_size: Optional[int] = None,
    dataset: str = "gittables",
) -> List[str]:
    

    #query_df = query_df.head(10)
    # Prepare positive correlation query
    labels, _ = get_labels_for_table(query_df, query_sketch_size)

    # Prepare negative correlation query
    query_df_anti = query_df.copy()
    query_df_anti.iloc[:, 1] = -query_df_anti.iloc[:, 1]
    labels_anti, _ = get_labels_for_table(query_df_anti, query_sketch_size)

    joined_labels = ", ".join([f"'{label}'" for label in labels])
    joined_labels_anti = ", ".join([f"'{label}'" for label in labels_anti])

    # Execute query
    start = time.perf_counter()
    cursor.execute(
        f"""
        (SELECT tableid_catcol_numcol, COUNT(*) as qcr_score, COUNT(*) as abs_qcr_score
        FROM {dataset}_qcr_index WHERE term IN ({joined_labels})
        GROUP BY tableid_catcol_numcol)
        UNION
        (SELECT tableid_catcol_numcol, -COUNT(*) as qcr_score, COUNT(*) as abs_qcr_score
        FROM {dataset}_qcr_index WHERE term IN ({joined_labels_anti})
        GROUP BY tableid_catcol_numcol)

        order by abs_qcr_score desc LIMIT {result_size * 2}
        """
    )
    result = cursor.fetchall()
    end = time.perf_counter()

    # Pick the top k unique results for tableid_catcol_numcol
    # Keeping the order of the query
    unique_results = []
    seen = []
    for tableid_catcol_numcol, qcr_score, abs_qcr_score in result:
        if tableid_catcol_numcol in seen:
            continue
        unique_results.append((tableid_catcol_numcol, abs_qcr_score))
        seen.append(tableid_catcol_numcol)
        if len(unique_results) == result_size:
            break
    fetch_time = end - start
    return unique_results, None, fetch_time


import hashlib
import heapq
import pandas as pd
import pickle
import textwrap
from collections import defaultdict, Counter
from pathlib import Path
from typing import Callable, Tuple, List, DefaultDict, Set, Union, Any, Dict
from unicodedata import numeric
import os

def callback_qcr(df_in: pd.DataFrame, part: any) -> pd.DataFrame:
    #print("+++++++++++++++++++++++++++++++++++")
    pid=os.getpid()
    #print(f"[DEBUG] Process {pid}: Input DataFrame retrieved for part: '{part}'")
    c_col = get_kc(df_in)
    #print(f"finished step 1 for part '{part}'")
    #print("The cat columns are:")
    n_col = get_c(df_in)
    #print(f"finished step 2 for part '{part}'")
    #print("The num columns are:")
    cross_product_tables_list = cross_product_tables(c_col, n_col, df_in.columns.name)
    #print(f"finished step 3 for part '{part}'")
    list1, list2 = [], []
    for i in cross_product_tables_list:
        print(i.shape)
        sketch = create_sketch(i.iloc[:, 0], i.iloc[:, 1], hash_md5, n=256)
        labels = key_labeling(sketch, hash_md5, inner_hash=False)
        list1.extend(labels)
        list2.extend([i.columns.name] * len(labels))
    #print(f"finished step 4 for part '{part}'")
    df_out = pd.DataFrame(zip(list1, list2), columns=['term', 'tableid_catcol_numcol'])
    print(f"[DEBUG] Process {pid}: Output DataFrame retrieved for part: '{part}'")
    print(f"[DEBUG] Process {pid}: Output DataFrame shape: {df_out.shape}")
    print(f"[DEBUG] Process {pid}: Output DataFrame columns: {df_out.columns}")    
    print("+++++++++++++++++++++++++++++++++++")

    return df_out


#############################################
# Paper related code                        #
#############################################

def hash_md5(obj: object) -> int:
    """
    Hashes an object to an integer value
    :param obj: Object that needs to be hashed
    :return: hashed value
    """
    hash_val = int.from_bytes(
        hashlib.md5(str(obj).encode("utf-8")).digest(), "big", signed=False
    )
    #print("hash value is:")
    #print(hash_val)
    return hash_val


def create_sketch(
        kc: List[str],
        c: List[numeric],
        hash_funct: Callable[[str], int],
        n=100
) -> List[Tuple[str, numeric]]:
    """
    This function creates a sketch of size n from two columns (one with numeric values, one with categorical values).
    It hashes the categorical column and builds a table (list of tuples) with the hashes and the corresponding values
    from the numerical column. This table is sorted by the hash-column and the rows with the n-smallest hash-values are
    kept for form the sketch
    :param kc: list of categorical keys (key column)
    :param c: list of numeric values (value column)
    :param hash_funct: collision free hash function string -> int/float
    :param n: size of sketch, default 100
    :return: sketch of size n for given columns
    """
    grouped = pd.DataFrame({'kc': kc, 'c': c}).groupby('kc').mean(numeric_only=True).reset_index()
    #print("grouped")
    #print(grouped)
    grouped = grouped.dropna()

    sketch = heapq.nsmallest(n, zip(grouped["kc"], grouped["c"]), key=lambda x: hash_funct(x[0]))
    return sketch


def key_labeling(sketch: List[Tuple[str, numeric]], h: Callable[[str], int] = lambda x: int.from_bytes(str(x).encode("utf-8"), "big", signed=False), inner_hash: bool = False) \
        -> List[Union[int, str]]:
    """
    labels keys according to their values' distribution. +key or -key
    :param sketch: table with keys and their values
    :param h: hash function if hashed keys shall be labeled, nothing if literal keys shall be labeled
    :return: returns a two col table of labeled keys and values
    """
    if not sketch:
        return []

    mue = sum([value for key, value in sketch]) / len(sketch)
    return [format(h(f'{f"{h(key):032x}" if inner_hash else key}{"+1" if value > mue else "-1"}'), "032x") for key, value in sketch]


def add_to_inverted_index(
        inverted_index: DefaultDict[int, Set[str]], terms: List[Union[int, str]], value: str
) -> None:
    """
    Add labeled key to inverted index, the table_id it originates from is the value of the dictionary entry.
    :param inverted_index: inverted index to add table content to
    :param terms: list of labeled keys from sketch
    :param value: name of table
    :return: none. Dictionary is edited
    """
    for key in terms:
        inverted_index[key].add(value)

def get_labels_for_table(df_in: pd.DataFrame, sketch_size: int) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
    """
    Creates a sketch for each column in the given table and labels the keys in the sketch.
    :param df_in: table to be labeled
    :param sketch_size: size of sketch per column
    :return: list of labeled keys and list of corresponding column names
    """
    c_col = get_kc(df_in)
    #print(c_col)
    n_col = get_c(df_in)
    #print(n_col)
    cross_product_tables_list = cross_product_tables(c_col, n_col, df_in.columns.name)
    #print(cross_product_tables_list)
    list1, list2 = [], []
    for i in cross_product_tables_list:
        sketch = create_sketch(i.iloc[:, 0], i.iloc[:, 1], hash_md5, n=sketch_size)
        labels = key_labeling(sketch, hash_md5, inner_hash=False)
        #print("labels are:")
        #print(labels)
        list1.extend(labels)
        list2.extend([i.columns.name] * len(labels))
    
    return list1, list2


#############################################
# Utils                                     #
#############################################

def load_index() -> DefaultDict[int, Set[str]]:
    """
    load index from disc if it exists. creates new, empty index otherwise
    :return: inverted index of type dictionary.
    """
    try:
        with open("index.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return defaultdict(set)


def save_index(index: DefaultDict[int, Set[str]]) -> None:
    """
    stores index on disc
    :param index: index of type dictionary
    :return: none. Stores on disc
    """
    with open("index.pickle", "wb") as f:
        pickle.dump(index, f)


def load_tables(folder_name: str) -> List[pd.DataFrame]:
    """
    Loads tables as pandas dataframe from all csv files in the folder
    :param folder_name: String of relative path from this script to the folder
    :return: tables (dataframes) contained in given folder
    """
    tables = []
    for path in Path(folder_name).glob("*.csv"):
        table = pd.read_csv(path, sep=";")
        table.columns.name = path.stem
        tables.append(table)

    return tables


def load_query() -> pd.DataFrame:
    """
    # Loads query table as pandas dataframe from csv file
    :return: query table as dataframe. 2 columns only!
    """

    return pd.read_csv("../../data/toy_data/A_0.csv", sep=";")


def cross_product_tables(cat_col: DefaultDict[str, List[str]], num_col: DefaultDict[str, List[numeric]],
                         table_id: str) -> List[pd.DataFrame]:
    """
    combines all numerical and categorical columns like a cross-product.
    eg: c1, c2 x n1, n2, n3 -> ['c1_n1', c1_n2', 'c1_n3','c2_n1', 'c2_n2', c2_n3']
    :param cat_col: default dict with column name as key and list of categorical-column-values as value.
    :param num_col: default dict with column name as key and list of numerical-column-values as value.
    :param table_id: name of table, that is to be split
    :return: list of named tables.
    """
    tables = []
    for cat_header in cat_col:
        for num_header in num_col:
            table = pd.DataFrame(list(zip(cat_col[cat_header], num_col[num_header])), columns=[cat_header, num_header])
            table.columns.name = f"{table_id}_|_{cat_header}_|_{num_header}"  # here we use the column names as name for the new table
            tables.append(table)
    return tables


def get_kc(table: pd.DataFrame) -> DefaultDict[str, List[str]]:
    """
    extract categorical columns from dataframe
    :param table: input table
    :return: dict of categorical columns by column name
    """
    kc_column_name = table.select_dtypes(include=["object"]).columns
    columns = defaultdict(List[str])
    #l=0
    for col in kc_column_name:
        columns[table.columns.to_list().index(col)] = (table[col].astype(str).apply(lambda x: x.strip().lower())).values.tolist()
        #l+=1
    #print(l)
    return columns


def get_c(table: pd.DataFrame) -> DefaultDict[str, List[numeric]]:
    """
    extract numerical columns from dataframe
    :param table: input table
    :return: dict of numerical columns by column name
    """
    c_column_name = table.select_dtypes(include=["float64", "int64"]).columns
    columns = defaultdict(List[str])
    #l=0
    for col in c_column_name:
        columns[table.columns.to_list().index(col)] = (table[col].values.tolist())
        #l+=1
    #print(l)
    return columns


def get_table_id(table: pd.DataFrame) -> str:
    """
    extract name from pandas dataFrame
    :param table: pandas dataFrame
    :return: name (string)
    """
    return table.columns.name


def print_dict(dictionary: DefaultDict[Any, List[Any]], identifier="") -> None:
    """
    pretty print version for dictionaries with string or numeric values
    :param dictionary: dict with list of String or list of numeric values
    :param identifier: a title for the dictionary that follows
    :return: none. prints to the console
    """
    if identifier != "":
        print(f"{identifier}:")

    for key in dictionary:
        if isinstance(dictionary[key], str):
            value = ', '.join(dictionary[key])
        else:
            value = str(dictionary[key]).strip('[]')

        prefix = f"{key} "
        wrapper = textwrap.TextWrapper(initial_indent=prefix, width=70,
                                       subsequent_indent=' ' * len(prefix))
        print(wrapper.fill(f"{value}"))
