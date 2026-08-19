import collections
import math
import numpy as np
import pandas as pd
from operator import itemgetter
from typing import Dict
from .DBHandler import DB_handler
from .qcr_main import search_qcr


class DataDiscovery_CLS:
    def __init__(self, db_name ='postgres', main_table='main_tokenized', conn_dict = None):
        self.DB = DB_handler(db_name, main_table, conn_dict)
        self.db_name = db_name
        self.main_table = main_table

    def clean_value_collection(self, values):
        return [str(v).replace("'", "''").strip() for v in values if str(v).lower() != 'nan']

    def create_sql_where_condition_from_value_list(self, values):
        return "'{}'".format("' , '".join(set(values)))

    def create_sql_where_condition_from_numerical_value_list(self, values):
        values = [str(x) for x in values]
        return "{}".format(" , ".join(values))


    def validateTable(self, queryResult, tableResult):
        queryTable = list()
        tableTable = list()
        for item in queryResult:
            queryTable.append(item[0])
        for item in tableResult:
            tableTable.append(item[0])

        validTable = list()
        for item in queryTable:
            if (item in tableTable):
                validTable.append(item)
        return validTable


    def evaluate_rows(self, input_row, col_dict, query_columns):
        vals = list(col_dict.values())
        query_cols_arr = np.array(query_columns)
        query_degree = len(query_cols_arr)
        matching_column_order = ''
        for q in query_cols_arr[-(query_degree - 1):]:
            q_index = list(query_columns).index(q)
            if input_row[q_index] not in vals:
                return False, ''
            else:
                for colid, val in col_dict.items():
                    if val == input_row[q_index]:
                        matching_column_order += '_{}'.format(str(colid))
        return True, matching_column_order


    def COCOA_correlation_discovery(self, query_table, join_column_name='query', target_column_name='target', k = 100):
        total_execution_time = 0
        total_fetch_time = 0

        k_t = 100
        k_c = k

        # print(self.con.execute('SELECT * FROM COCOA WHERE table_col_id = \'32364_7\';').fetchall())
        # print(self.con.execute('SELECT * FROM AllTables WHERE TableId = 32364 AND ColumnId = 7;').fetchall())
        # exit(0)
        pd.set_option('display.max_columns', None)
        value_iter = [str(v).replace("'", "''").strip() for v in query_table[join_column_name]]
        query_condition = "'{}'".format("' , '".join(value_iter))
        # content_index_df = self.con.execute("""SELECT * FROM (SELECT CONCAT(TableId, \'_\', ColumnId) AS table_column FROM AllTables
        # WHERE CellValue IN ({})
        # GROUP BY TableId, ColumnId
        # ORDER BY COUNT(*) DESC LIMIT {}) AS JOINABLE_TABLES
        # INNER JOIN COCOA ON JOINABLE_TABLES.table_column = COCOA.table_col_id;""".format(query_condition, k_t)).fetch_df()

        query_table['rank_target'] = query_table[target_column_name].rank(method='average')
        target_ranks = np.array(query_table['rank_target'])
        std_target_rank = np.std(target_ranks)
        target_rank_sum = sum(target_ranks)

        # -----------------------------------------------------------------------------------------------------------
        # FINDING JOINABLE COLUMNS
        # -----------------------------------------------------------------------------------------------------------
        overlap_columns, execution_time, fetch_time = [x[0] for x in self.DB.execute_and_fetchall("""SELECT CONCAT(CONCAT(TableId, \'_\'), ColumnId) FROM AllTables 
        WHERE CellValue IN ({}) GROUP BY TableId, ColumnId ORDER BY COUNT(*) DESC LIMIT {};""".format(query_condition, k_t))]
        total_execution_time += execution_time
        total_fetch_time += fetch_time
        table_ids = []
        column_ids = []
        for o in overlap_columns:
            table_ids.append(int(o.split('_')[0]))
            column_ids.append(int(o.split('_')[1]))

        # -----------------------------------------------------------------------------------------------------------
        # FETCHING CONTENT OF TABLES CONTAINING JOINABLE COLUMNS
        # -----------------------------------------------------------------------------------------------------------

        joint_table_ids = '\',\''.join(overlap_columns)
        joinable_content, execution_time, fetch_time = self.DB.execute_and_fetchall(f'SELECT CONCAT(CONCAT(TableId, \'_\'), ColumnId), CellValue, RowId '
                           f'FROM AllTables '
                           f'WHERE CONCAT(CONCAT(TableId, \'_\'), ColumnId) IN (\'{joint_table_ids}\') '
                           f'ORDER BY CONCAT(CONCAT(TableId, \'_\'), ColumnId), RowId;')
        total_execution_time += execution_time
        total_fetch_time += fetch_time

        external_joinable_tables = pd.DataFrame(joinable_content, columns=['table_col_id', 'tokenized', 'rowid'])
        joinable_tables_dict = {}  # store content of tables containing at least one column that is joinable with query
        for table_col_id, group in external_joinable_tables.groupby(['table_col_id']):
            keys = list(group['tokenized'])
            values = list(group['rowid'])
            item = dict(zip(keys, values))
            joinable_tables_dict[table_col_id] = item

        # -----------------------------------------------------------------------------------------------------------
        # INDEX PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        max_column_ids, execution_time, fetch_time = self.DB.execute_and_fetch_df('SELECT TableId AS tableid, MAX(ColumnId) AS max_col_id FROM AllTables WHERE TableId IN (\'{}\') GROUP BY TableId;'
                                       .format('\',\''.join([str(x) for x in table_ids])), ['tableid', 'max_col_id'])
        total_execution_time += execution_time
        total_fetch_time += fetch_time
        # Now we compute all table_col_ids for which we need to fetch the index
        max_column_dict = max_column_ids.astype(int).set_index('tableid').to_dict()['max_col_id']
        table_col_ids = []
        for table_id in max_column_dict:
            for i in range(max_column_dict[table_id] + 1):
                table_col_ids.append(str(table_id) + '_' + str(i))

        # Datastructures in which we store the index for each table_col_id
        order_dict = {}
        binary_dict = {}
        min_dict = {}
        numerics_dict = {}

        joint_table_column_ids = '\',\''.join(table_col_ids)
        order_index, execution_time, fetch_time = self.DB.execute_and_fetchall(f'SELECT table_col_id, is_numeric, min_index, order_list, binary_list '
                           f'FROM COCOA '
                           f'WHERE table_col_id IN (\'{joint_table_column_ids}\');')
        total_execution_time += execution_time
        total_fetch_time += fetch_time
        cocoa_index = pd.DataFrame(order_index, columns=['table_col_id', 'is_numeric', 'min_index', 'order_list', 'binary_list'])

        for _, index in cocoa_index.iterrows():
            table_col_id = index['table_col_id']
            order_dict[table_col_id] = [int(x) for x in index['order_list'].split(',')]
            binary_dict[table_col_id] = [x for x in index['binary_list'].split(',')]
            min_dict[table_col_id] = int(index['min_index'])
            numerics_dict[table_col_id] = bool(index['is_numeric'])

        # -----------------------------------------------------------------------------------------------------------
        # PREPARATION
        # -----------------------------------------------------------------------------------------------------------
        input_size = len(query_table)
        column_name = []
        column_correlation = []
        join_maps = {}

        # -----------------------------------------------------------------------------------------------------------
        # CORRELATION CALCULATION
        # -----------------------------------------------------------------------------------------------------------

        def generate_join_map(col: pd.Series, column: Dict) -> np.ndarray:

            vals = column.values()
            vals = [int(x) for x in vals]
            join_table = np.full(max(vals) + 1, -1)

            q = np.array(col)
            for i in np.arange(len(q)):
                x = q[i]
                index = column.get(x, -1)
                if index != -1:
                    join_table[index] = i

            return join_table


        for i in np.arange(len(table_ids)):
            column = column_ids[i]
            table = table_ids[i]
            max_col = max_column_dict[table]

            joinMap = generate_join_map(query_table[join_column_name], joinable_tables_dict[str(table) + '_' + str(column)])

            for c in np.arange(max_col + 1):
                if c == column:
                    continue

                t_c_key = f'{table}_{c}'
                join_maps[t_c_key] = joinMap

                if t_c_key not in numerics_dict:
                    continue
                is_numeric_column = numerics_dict[t_c_key]
                pointer = min_dict[t_c_key]
                order_index = order_dict[t_c_key]
                binary_index = binary_dict[t_c_key]

                query_table['new_external_rank'] = math.ceil(input_size / 2)
                external_rank = query_table['new_external_rank'].values

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
                        if binary_index[pointer] == '1':
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
                        pointer = int(order_index[pointer])
                    cor = np.corrcoef(query_table['rank_target'], external_rank)[0, 1]

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
                        pointer = int(order_index[pointer])
                    cor = max_correlation
                column_name += [t_c_key]
                column_correlation += [cor]

        # Now we get the topk columns with highest correlation
        overall_list = []
        for i in np.arange(len(column_correlation)):
            overall_list += [[column_correlation[i], column_name[i]]]
        sorted_list = sorted(overall_list, key=itemgetter(0), reverse=True)

        topk_table_col_ids = []
        for important_column_index in np.arange(min(k_c, len(sorted_list))):
            important_column = sorted_list[important_column_index]
            topk_table_col_ids += [important_column[1]]
        if 'new_external_rank' in query_table:
            query_table = query_table.drop('new_external_rank', axis=1)

        return topk_table_col_ids
    

    def qcr_correlation_search(self, df, categorical, numerical, k, db_con=None):
        if self.main_table.endswith("_qcr_index"):
            dataset = self.main_table[: -len("_qcr_index")]
        elif self.main_table.startswith("main_tokenized"):
            dataset = "main_tokenized"
        elif self.main_table.startswith("git"):
            dataset = "gittables"
        elif self.main_table.startswith("nyc"):
            dataset = "nyc"
        elif self.main_table.startswith("canada"):
            dataset = "cuk"
        elif self.main_table.startswith("cuk"):
            dataset = "cuk"
        elif self.main_table.startswith("synthetic_table_corpus"):
            dataset = "synthetic_table_corpus"
        elif self.main_table.startswith("santos_small"):
            dataset = "santos_small"
        else:
            raise Exception("QCR does not support this dataset")
        
        df = df.copy()[[categorical, numerical]]
        
        if db_con is not None:
            return search_qcr(df, db_con, k, 256, dataset)
        
        return search_qcr(df, self.DB.postgres_con, k, 256, dataset)
