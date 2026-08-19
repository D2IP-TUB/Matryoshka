import logging
import re
import math
from operator import itemgetter
import pandas as pd
import numpy as np
import warnings


def get_cleaned_text(text):
    """
    Removes stop words and formats text in ascii format.

    :param text: Given text
    :return: CLeaned text
    """
    if text is None:
        return ''
    stopwords = ['a', 'the', 'of', 'on', 'in', 'an', 'and', 'is', 'at', 'are', 'as', 'be', 'but', 'by', 'for', 'it',
                 'no', 'not', 'or'
        , 'such', 'that', 'their', 'there', 'these', 'to', 'was', 'with', 'they', 'will', 'v', 've', 'd']

    cleaned = re.sub('[\W_]+', ' ', text.encode('ascii', 'ignore').decode('ascii'))
    feature_one = re.sub(' +', ' ', cleaned).strip()

    for x in stopwords:
        feature_one = feature_one.replace(f' {x} ', ' ')
        if feature_one.startswith(f'{x} '):
            feature_one = feature_one[len(f'{x} '):]
        if feature_one.endswith(f' {x}'):
            feature_one = feature_one[:-len(f' {x}')]
    return feature_one


def create_index(values):
    """
    Creates order index for given column/list of values as presented in the COCOA paper.
    Index consists of:
    - min_index: Index of minimum in order_list and binary_list
    - order_list: List of indexes from which the ranks can be constructed in linear time.
    - binary_list: List of boolean values, True if current and next value are equal


    :param values: Input column (list).
    :return: min_index, order_list, binary_list
    """
    def is_numeric(s):
        """
        Checks if given value is numeric.

        :param s: Value
        :return: True, if value is numeric
        """
        if s.lower() == 'nan':
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_numeric_list(l):
        """
        Checks if given list is numeric.

        :param l: list
        :return: True, if all values in the list are numeric.
        """
        for i in np.arange(len(l)):
            if l[i] is None or l[i] == '':
                l[i] = np.nan
            else:
                l[i] = l[i]

        result = [s for s in l if is_numeric(str(s))]
        return len(result) == len(l)

    rows = np.arange(0, len(values), 1)
    if is_numeric_list(values):
        for i in np.arange(len(values)):
            if values[i] is None or values[i] == '':
                values[i] = np.nan
        values = [float(i) for i in values]
    else:
        for i in np.arange(len(values)):
            if values[i] is None:
                values[i] = ''
        values = [str(i) for i in values]
    ranks = list(pd.Series(values).rank(na_option='bottom', method='average'))

    rows_sorted_based_on_ranks = [x for _, x in sorted(zip(ranks, rows))]
    min_index = rows_sorted_based_on_ranks[0]  # starting point in the order index
    order_list = np.empty(len(rows), dtype=int)
    binary_list = np.empty(len(rows), dtype=str)
    sorted_ranks = np.sort(ranks).copy()
    for i in np.arange(len(rows) - 1):
        order_list[i] = rows_sorted_based_on_ranks[i + 1]
        # if both values are NaN we treat them equally
        if np.isnan(sorted_ranks[i]) and np.isnan(sorted_ranks[i + 1]) or sorted_ranks[i] == sorted_ranks[
            i + 1]:
            binary_list[i] = 'F'
        else:
            binary_list[i] = 'T'
    order_list[len(rows) - 1] = -1  # Maximum value
    binary_list[len(rows) - 1] = -1  # Maximum value

    final_order_list = [x for _, x in
                        sorted(zip(rows_sorted_based_on_ranks, order_list))]  # order list in the order index
    final_binary_list = [x for _, x in
                         sorted(zip(rows_sorted_based_on_ranks, binary_list))]  # binary list in the order index

    return min_index, final_order_list, final_binary_list


class COCOAHandler:
    def __init__(self, conn, db_tables):
        """
        Creates a COCOA instance to enrich given datasets with external data.

        :param conn: Database connection, e.g. Vertica, Postgres, ...
        :param db_tables: DB table names. Format: { dt: <table_name>, mt: <table_name>, oi: <table_name>,
                                                    mc: <table_name> }
        """
        self.conn = conn
        self.logging = logging

        # Parse DB tables
        try:
            self.dt_table = db_tables['dt']
            self.mt_table = db_tables['mt']
            self.oi_table = db_tables['oi']
            self.mc_table = db_tables['mc']
        except KeyError:
            raise ValueError('COCOA: Invalid constructor input for parameter db_tables.')

    def enrich(self, data, k_c, k_t, query_column='query', target_column='target'):
        """

        :param data: Input dataset, contains query and target columns.
        :param k_c: Number of columns to return.
        :param k_t: Number of overlap columns to compute the
        :param query_column: Name of the query column in the given pandas dataframe
        :param target_column: Name of the target column in the given pandas dataframe
        :return: Given dataframe with joint external data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            def generate_rank(col):
                """
                Returns rank of the given column.

                :param col: Pandas dataframe column
                :return: Pandas dataframe column containing rank for each row
                """
                return col.rank(na_option='bottom', method='average')

            def generate_join_map(q_col, dict):
                """
                Generates a JoinMap for a given column

                :param query_column: Column
                :param dict: Dict containing joinable columns
                :return: JoinMap
                """
                vals = dict.values()
                vals = [int(x) for x in vals]
                join_table = np.full(max(vals) + 1, -1)

                q = np.array(q_col)
                for i in np.arange(len(q)):
                    x = q[i]
                    index = dict.get(x, -1)
                    if index != -1:
                        join_table[index] = i

                return join_table

            logging.info('=== Starting COCOA ===')

            # Query preparation
            dataset = data.copy()
            dataset = dataset.apply(lambda x: x.astype(str).str.lower())
            dataset[query_column] = dataset[query_column].apply(get_cleaned_text)

            # Target preparation
            dataset['rank_target'] = generate_rank(dataset[target_column])
            target_ranks = np.array(dataset['rank_target'])
            std_target_rank = np.std(target_ranks)
            target_rank_sum = sum(target_ranks)

            # calculate joinable columns
            logging.info('Finding joinable columns...')
            distinct_clean_values = dataset[query_column].unique()
            joint_distinct_values = '\',\''.join(distinct_clean_values)

            overlap_query = f'SELECT ol.table_col_id, number_of_tokens ' \
                            f'FROM (' \
                            f'    SELECT table_col_id, COUNT(tokenized) as number_of_tokens ' \
                            f'    FROM {self.dt_table} ' \
                            f'    WHERE tokenized IN (\'{joint_distinct_values}\') ' \
                            f'    GROUP BY table_col_id' \
                            f') as ol ' \
                            f'ORDER BY number_of_tokens DESC ' \
                            f'LIMIT {k_t};'

            overlap_columns = list(pd.read_sql(overlap_query, self.conn)['table_col_id'])
            pd.read_sql(overlap_query, self.conn).to_parquet('overlap_index.parquet')
            logging.info('Finished.')

            if not overlap_columns:
                logging.info('No joinable columns found.')
                logging.info('=== Finished COCOA ===')
                return dataset     # no external content added

            # Extract table and column ids from each
            table_ids = []
            column_ids = []
            for o in overlap_columns:
                table_ids.append(int(o.split('_')[0]))
                column_ids.append(int(o.split('_')[1]))
            joint_overlap_columns = '\',\''.join(overlap_columns)

            logging.info('Fetching content of joinable columns...')

            token_query = f'SELECT table_col_id, tokenized, rowid ' \
                        f'FROM {self.mt_table} ' \
                        f'WHERE table_col_id IN (\'{joint_overlap_columns}\') ' \
                        f'ORDER BY table_col_id, rowid;'

            external_joinable_tables = pd.read_sql(token_query, self.conn)
            external_joinable_tables.to_parquet('token_index.parquet')
            joinable_tables_dict = {}   # store content of tables containing at least one column that is joinable with query

            for name, group in external_joinable_tables.groupby(['table_col_id']):
                keys = list(group['tokenized'])
                values = list(group['rowid'])
                item = dict(zip(keys, values))
                joinable_tables_dict[name[0]] = item

            logging.info('Finished.')
            logging.info('Fetching number of columns for each table...')

            # We need the number of cols for all table ids that contain at least one overlapping column
            joint_table_ids = '\',\''.join([str(i) for i in table_ids])
            max_col_query = f'SELECT tableid, max_colid ' \
                            f'FROM {self.mc_table} ' \
                            f'WHERE tableid IN (\'{joint_table_ids}\');'

            max_column_ids = pd.read_sql(max_col_query, self.conn)
            logging.info('Finished.')

            # Now we compute all table_col_ids for which we need to fetch the index
            max_column_dict = max_column_ids.astype(int).set_index('tableid').to_dict()['max_colid']
            table_col_ids = []
            for table_id in max_column_dict:
                for i in range(max_column_dict[table_id] + 1):
                    table_col_ids.append(str(table_id) + '_' + str(i))

            logging.info('Fetching order index...')
            joint_table_col_ids = '\',\''.join(table_col_ids)
            index_query = f'SELECT table_col_id, is_numeric, min_index, order_list, binary_list ' \
                        f'FROM {self.oi_table} ' \
                        f'WHERE table_col_id IN (\'{joint_table_col_ids}\')'

            index = pd.read_sql(index_query, self.conn)
            index.to_parquet('order_index.parquet')
            logging.info('Finished.')

            # Datastructures in which we store the index for each table_col_id
            order_dict = {}
            binary_dict = {}
            min_dict = {}
            numerics_dict = {}

            for _, row in index.iterrows():
                table_col_id = row['table_col_id']
                order_dict[table_col_id] = [int(i) for i in row['order_list'].split(',')]
                binary_dict[table_col_id] = [str(i) for i in row['binary_list'].split(',')]
                min_dict[table_col_id] = int(row['min_index'])
                numerics_dict[table_col_id] = bool(row['is_numeric'])

            # Prepare results and calculation
            input_size = len(dataset)
            column_name = []
            column_correlation = []
            join_maps = {}

            logging.info('Calculating correlations...')
            for i in np.arange(len(table_ids)):
                column = column_ids[i]
                table = table_ids[i]
                max_col = max_column_dict[table]
                joinMap = generate_join_map(dataset[query_column], joinable_tables_dict[str(table) + '_' + str(column)])

                for c in np.arange(max_col + 1):
                    if c == column:
                        continue

                    t_c_key = f'{table}_{c}'
                    join_maps[t_c_key] = joinMap

                    is_numeric_column = numerics_dict[t_c_key]
                    pointer = min_dict[t_c_key]
                    order_index = order_dict[t_c_key]
                    binary_index = binary_dict[t_c_key]

                    dataset['new_external_rank'] = math.ceil(input_size / 2)
                    external_rank = dataset['new_external_rank'].values

                    # We use the order index to compute the ranks of each column
                    if is_numeric_column:
                        # Spearman correlation coefficient
                        counter = 1
                        jump_flag = False
                        current_counter_assigned = False

                        equal_values = np.empty(len(order_index), dtype=np.int64)
                        equal_values_count = 0

                        # Average-rank for equal values:
                        while pointer != -1:
                            if jump_flag and current_counter_assigned:
                                counter += 1
                                jump_flag = False
                                current_counter_assigned = False

                            input_index = joinMap[pointer]
                            if input_index != -1:
                                external_rank[input_index] = counter
                                current_counter_assigned = True

                            # T = value[i] = value[i + 1] in column
                            if binary_index[pointer] == 'T':
                                if equal_values_count:
                                    equal_values[equal_values_count] = pointer
                                    equal_values_count += 1

                                    # We count all equal values and assign average for each
                                    rank = 0
                                    for j in range(0, equal_values_count):
                                        rank += external_rank[joinMap[equal_values[j]]]
                                    rank = rank / equal_values_count

                                    for j in range(0, equal_values_count):
                                        external_rank[joinMap[equal_values[j]]] = rank
                                    equal_values_count = 0
                                jump_flag = True
                            else:
                                equal_values[equal_values_count] = pointer
                                equal_values_count += 1
                                counter += 1

                            # In the end, we have to check if the last values were equal and assign the average rank
                            if equal_values_count:
                                rank = 0
                                for j in range(0, equal_values_count):
                                    rank += external_rank[joinMap[equal_values[j]]]
                                rank = rank / equal_values_count

                                for j in range(0, equal_values_count):
                                    external_rank[joinMap[equal_values[j]]] = rank
                                equal_values_count = 0

                            pointer = order_index[pointer]
                        cor = np.corrcoef(dataset['rank_target'], external_rank)[0, 1]

                    else:
                        # Pearson correlation coefficient
                        max_correlation = 0
                        ohe_sum = 0
                        ohe_qty = 0
                        jump_flag = False

                        while pointer != -1:
                            if jump_flag:
                                if ohe_qty > 0:
                                    correlation = ((input_size * ohe_sum) - (ohe_qty * target_rank_sum)) / (
                                            std_target_rank * input_size * math.sqrt((ohe_qty * (input_size - ohe_qty))))
                                    if abs(correlation) > max_correlation:
                                        max_correlation = abs(correlation)
                                ohe_qty = 0
                                ohe_sum = 0
                                jump_flag = False

                            input_index = joinMap[pointer]
                            if input_index != -1:
                                ohe_sum += target_ranks[input_index]
                                ohe_qty += 1

                            if binary_index[pointer] == 'T':
                                jump_flag = True
                            pointer = order_index[pointer]
                        cor = max_correlation
                    column_name += [t_c_key]
                    column_correlation += [cor]

            logging.info('Finished.')

            # Now we get the topk columns with highest correlation
            overall_list = []
            for i in np.arange(len(column_correlation)):
                overall_list += [[column_correlation[i], column_name[i]]]
            sorted_list = sorted(overall_list, key=itemgetter(0), reverse=True)

            topk_table_col_ids = []
            for important_column_index in np.arange(min(k_c, len(sorted_list))):
                important_column = sorted_list[important_column_index]
                topk_table_col_ids += [important_column[1]]
            if 'new_external_rank' in dataset:
                dataset = dataset.drop('new_external_rank', axis=1)

            logging.info('Fetching content for top-k columns...')

            # We fetch the content only for the topk correlating columns
            table_col_ids = '\',\''.join(topk_table_col_ids)
            content_query = f'SELECT table_col_id, tokenized ' \
                            f'FROM {self.mt_table} ' \
                            f'WHERE table_col_id IN (\'{table_col_ids}\') ' \
                            f'ORDER BY table_col_id, rowid;'

            content_result = pd.read_sql(content_query, self.conn)
            logging.info('Finished.')

            content = {}
            for name, group in content_result.groupby(['table_col_id']):
                content[name] = list(group['tokenized'])

            # Materialize join
            for cor, name in sorted_list[:min(k_c, len(sorted_list))]:
                tokens = content[str(name)]
                join_map = join_maps[str(name)]
                joint_tokens = ['' for _ in range(len(dataset[query_column]))]
                for i in range(len(tokens)):
                    if join_map[i] != -1:
                        joint_tokens[join_map[i]] = str(tokens[i])

                dataset[name] = joint_tokens

        return dataset
