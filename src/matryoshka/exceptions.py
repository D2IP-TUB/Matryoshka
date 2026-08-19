import polars as pl


class KeyNotFoundError(Exception):
    '''
    Exception raised when at least one foreign key was not found in the index
    '''
    def __init__(self, query_column: pl.DataFrame):
        super().__init__(f'Failed to find foreign keys for query column {query_column} in the database')


class InvalidQueryColumnType(Exception):
    '''
    Exception raised when the query column type is not String
    '''
    def __init__(self, query_column: pl.DataFrame):
        super().__init__(f'Query column must be of type String but yours is of type {query_column.dtypes[0]}')


class UserTableNotProcessed(Exception):
    '''
    Exception raised when the user table contains non-numeric columns
    '''
    def __init__(self):
        super().__init__('User table must contain only numeric columns')


class UserTableHasDuplicates(Exception):
    '''
    Exception raised when the user table contains duplicate rows
    '''
    def __init__(self):
        super().__init__('User table contains duplicate rows')


class InvalidMatch(Exception):
    '''
    Exception raised when the match between token from index table and user table is not valid
    '''
    def __init__(self, token, base_table_token, table_row_index, table_column_index, base_table_row):
        super().__init__(f'Match {token} = {base_table_token} is not valid.\n Index table loc: {table_row_index} : {table_column_index}\n User table loc: {base_table_row}')


class LinAlgError(Exception):
    '''
    Exception raised when a linear algebra operation fails
    '''
    def __init__(self, cond_number: float):
        super().__init__(f'Linear algebra error: condition number {cond_number} is too high. Near singular matrix or ill-conditioned matrix.')


class EmptyAugmentation(Exception):
    '''
    Exception raised when no augmentation is possible
    '''
    def __init__(self):
        super().__init__('All augmentation candidates were skipped due to constant features or linear dependencies.')
