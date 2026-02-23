from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import bisect
import os
import torch
from functools import reduce
from torch.utils.data import DataLoader, TensorDataset
import psutil
import copy
from itertools import combinations
from ....utils.common import process_key


def cleanup(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            del arg
    torch.cuda.empty_cache()


def linear_regression_residuals(df, X_columns, Y_column, adjusted=False):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Ensure that X_columns exist in the dataframe
    if not all(item in df.columns for item in X_columns):
        raise ValueError('Not all specified X_columns are in the dataframe.')
    if Y_column not in df.columns:
        raise ValueError('The Y_column is not in the dataframe.')

    # Prepare the feature matrix X by selecting the X_columns and adding an intercept term
    X = df[X_columns].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term
    # Extract the target variable vector Y
    Y = df[Y_column].values

    # Calculate theta using the pseudo-inverse
    # theta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    # Make predictions
    # Y_pred = X @ theta
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    # Calculate residuals
    residuals = Y - Y_pred
    # Add residuals to the dataframe
    df['residuals'] = residuals

    # Calculate R-squared
    SS_res = (residuals ** 2).sum()
    SS_tot = ((Y - np.mean(Y)) ** 2).sum()
    R_squared = 1 - SS_res / SS_tot
    
    if adjusted:
        # Calculate Adjusted R-squared
        n = X.shape[0]  # Number of observations
        p = X.shape[1] - 1  # Number of predictors, excluding intercept
        R_squared = 1 - ((1 - R_squared) * (n - 1)) / (n - p - 1)
    return df, R_squared

class SketchLoader:
    def __init__(self, batch_size, device='cpu', disk_dir='sketches/', is_buyer=False):
        self.batch_size = batch_size
        self.sketch_1_batch = {}
        self.sketch_x_batch = {}
        self.sketch_x_x_batch = {}
        self.sketch_x_y_batch = {}
        self.is_buyer = is_buyer
        self.device = device
        self.num_batches = 0
        self.disk_dir = disk_dir

    def load_sketches(self, seller_1, seller_x, seller_x_x, feature_index_map, seller_id, 
                      cur_df_offset=0, to_disk=False, seller_x_y=None):
        
        if self.is_buyer:
            # Each buyer sketch will only have one column with respect to each join key
            # TODO: Now assume each buyer sketch is small
            if seller_x_y is not None:
                self.sketch_1_batch[0] = seller_1[:, 0:1].to(self.device)
                self.sketch_x_y_batch[0] = seller_x_y.to(self.device)
            else:
                self.sketch_1_batch[0] = seller_1.to(self.device)
            self.sketch_x_batch[0] = seller_x.to(self.device)
            self.sketch_x_x_batch[0] = seller_x_x.to(self.device)
            feature_index_map[0] = [(0, seller_id, 0)]
            return
        
        if not self.sketch_x_batch:
            # If the dictionary is empty, start with batch number 0
            self.sketch_1_batch[0] = seller_1[:, :min(
                self.batch_size, seller_1.size(1))]
            remaining_seller_1 = seller_1[:, self.batch_size:]
            self.sketch_x_batch[0] = seller_x[:, :min(
                self.batch_size, seller_x.size(1))]
            remaining_seller_x = seller_x[:, self.batch_size:]
            self.sketch_x_x_batch[0] = seller_x_x[:, :min(
                self.batch_size, seller_x_x.size(1))]
            remaining_seller_x_x = seller_x_x[:, self.batch_size:]
            feature_index_map[0] = [(0, seller_id, 0)]
            cur_df_offset = self.batch_size
        else:
            # Find the last batch number
            last_batch_num = max(self.sketch_x_batch.keys())
            last_batch_1 = self.sketch_1_batch[last_batch_num]
            last_batch_x = self.sketch_x_batch[last_batch_num]
            last_batch_x_x = self.sketch_x_x_batch[last_batch_num]

            # Calculate remaining space in the last batch
            remaining_space = self.batch_size - last_batch_x.size(1)

            # Append as much as possible to the last batch
            if remaining_space > 0:
                amount_to_append = min(remaining_space, seller_x.size(1))
                self.sketch_1_batch[last_batch_num] = torch.cat(
                    [last_batch_1, seller_1[:, :amount_to_append]], dim=1)
                self.sketch_x_batch[last_batch_num] = torch.cat(
                    [last_batch_x, seller_x[:, :amount_to_append]], dim=1)
                self.sketch_x_x_batch[last_batch_num] = torch.cat(
                    [last_batch_x_x, seller_x_x[:, :amount_to_append]], dim=1)
                remaining_seller_1 = seller_1[:, amount_to_append:]
                remaining_seller_x = seller_x[:, amount_to_append:]
                remaining_seller_x_x = seller_x_x[:, amount_to_append:]
                bisect.insort(feature_index_map[last_batch_num], (last_batch_x.size(
                    1), seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                # No space left in the last batch, start a new batch
                last_batch_num += 1
                # feature_index_map[last_batch_num] =
                self.sketch_1_batch[last_batch_num] = seller_1[:, :min(
                    self.batch_size, seller_1.size(1))]
                self.sketch_x_batch[last_batch_num] = seller_x[:, :min(
                    self.batch_size, seller_x.size(1))]
                self.sketch_x_x_batch[last_batch_num] = seller_x_x[:, :min(
                    self.batch_size, seller_x_x.size(1))]
                remaining_seller_1 = seller_1[:, self.batch_size:]
                remaining_seller_x = seller_x[:, self.batch_size:]
                remaining_seller_x_x = seller_x_x[:, self.batch_size:]
                feature_index_map[last_batch_num] = [
                    (0, seller_id, cur_df_offset)]
                cur_df_offset += self.batch_size
        self.num_batches = len(self.sketch_x_batch.keys())

        # Recursively append the remaining parts
        # if there is remaining part, that means the previous batch is occupied 
        if remaining_seller_x.size(1) > 0:           
            # Create the directory if it doesn't exist
            if not os.path.exists(self.disk_dir):
                os.makedirs(self.disk_dir)
            # Save the tensor
            if to_disk:
                prev_batch_id = self.num_batches-1
                sketch_1_batch = self.sketch_1_batch[prev_batch_id]
                sketch_x_batch = self.sketch_x_batch[prev_batch_id]
                sketch_x_x_batch = self.sketch_x_x_batch[prev_batch_id]

                torch.save(sketch_1_batch, os.path.join(self.disk_dir, "sketch_1_" + str(prev_batch_id) + ".pt"))
                torch.save(sketch_x_batch, os.path.join(self.disk_dir, "sketch_x_" + str(prev_batch_id) + ".pt"))
                torch.save(sketch_x_x_batch, os.path.join(self.disk_dir, "sketch_x_x_" + str(prev_batch_id) + ".pt"))
                del self.sketch_1_batch[prev_batch_id]
                del self.sketch_x_batch[prev_batch_id]
                del self.sketch_x_x_batch[prev_batch_id]
            self.load_sketches(remaining_seller_1, remaining_seller_x, remaining_seller_x_x,
                               feature_index_map, seller_id, cur_df_offset) 
            
    def get_sketches(self, batch_id, from_disk=False):
        sketch_x_y_batch = None
        if from_disk:
            # Buyer dataset never on disk
            sketch_1_batch = torch.load(os.path.join(self.disk_dir, 
                                                     "sketch_1_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_batch = torch.load(os.path.join(self.disk_dir, 
                                                     "sketch_x_" + str(batch_id) + ".pt")).to(self.device)
            sketch_x_x_batch = torch.load(os.path.join(self.disk_dir, 
                                                       "sketch_x_x_" + str(batch_id) + ".pt")).to(self.device)
        else:
            sketch_1_batch = self.sketch_1_batch[batch_id].to(self.device)
            sketch_x_batch = self.sketch_x_batch[batch_id].to(self.device)
            sketch_x_x_batch = self.sketch_x_x_batch[batch_id].to(self.device)
            if batch_id in self.sketch_x_y_batch:
                sketch_x_y_batch = self.sketch_x_y_batch[batch_id].to(self.device)
        return sketch_1_batch, sketch_x_batch, sketch_x_x_batch, sketch_x_y_batch

    def get_num_batches(self):
        return self.num_batches
    
class SketchBase:
    def __init__(self, join_key_domain, device='cpu', is_buyer=False):
        self.feature_index_mapping = {}
        self.dfid_feature_mapping = {}
        self.device = device
        self.join_key_domain = join_key_domain
        self.current_df_id = 0
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_free_mem = gpu_total_mem - torch.cuda.memory_allocated(0)
        else:
            self.gpu_free_mem = None
        self.gpu_batch_size, self.ram_batch_size = self.estimate_batch_size()    
        # TODO: join key domain requried in estimate_batch_size
        # sketch loader only needs to fully utilize gpu memory
        self.sketch_loader = SketchLoader(self.gpu_batch_size, device=device, is_buyer=is_buyer)


    """
    This function is used to estimate the batch size based on the available memory.
    It will return the batch size for both GPU and RAM.

    @param join_key_domain: a dictionary containing the domain of each join key. 
            We need this because the size(0) would be the product of the domain of all join keys.
    
    @return: a tuple containing the batch size for GPU and RAM

    """
    def estimate_batch_size(self):
        # Similar logic as search_gpu.py, just copy and paste
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), 
                              self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        # TODO: 2 is a workaround
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 3 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 3 * tensor_width)
        return gpu_batch_size, ram_batch_size

    """
    This function is used to register a dataframe to the sketch base.
    As a base function, it only takes in 1, x, x_x, feature_num, df_id, and offset as input params. 
    It will check if the current tensors satisfies the ram requirements and load it to sketches with load_sketches. 
    After loading the sketches, it returns the updated offset corresponding to the id of the dataframe.

    @param df_id: the unique identifier of the dataframe. Later on, when we want to fetch the 
              corresponding df_id, we could use this identifier. It stores in a priority queue in feature_index_mapping.
    @param feature_num: the number of features in the dataframe
    @param seller_1: the 1 matrix of the dataframe
    @param seller_x: the X matrix of the dataframe
    @param seller_x_x: the X_X matrix of the dataframe
    @param to_disk: whether to save the sketches to disk

    @return: a dictionary containing the batch_id, df_id, and offset
    """
    def _register_df(self, df_id, feature_num, seller_1, seller_x, seller_x_x, seller_x_y=None, to_disk=False):
        # Before loading the sketches, check if the current tensors satisfy the ram requirements
        # if seller_x.size(1) > min(self.gpu_batch_size, self.ram_batch_size):
        #     raise ValueError("The number of features in the dataframe is too large.")
        # Load the sketches
        self.sketch_loader.load_sketches(
            seller_1 = seller_1,
            seller_x = seller_x,
            seller_x_x = seller_x_x,
            seller_x_y = seller_x_y,
            feature_index_map = self.feature_index_mapping,
            seller_id = df_id,
            to_disk = to_disk
        )
        # Return the updated offset corresponding to the df_id. This is not efficient. Only for unit test usage.
        def find_by_seller_id(feature_index_map, seller_id):
            for batch_id, entries in feature_index_map.items():
                for end_pos, id, offset in entries:
                    if id == seller_id:
                        return batch_id, offset
            return None, None  # If the seller_id is not found

        batch_id, offset = find_by_seller_id(self.feature_index_mapping, df_id)
        return {"batch_id": batch_id, "df_id": df_id, "offset": offset}



    """
    This function is used to get the 1, X, X_X, and X_Y matrices of a dataframe.
    takes in a df( with join key as a col)  and the key_domainsreturn and returns the calibrated 1, x, x_x tensors.
    @param df_id: the unique identifier of the dataframe.
    @param df: the dataframe to be calibrated. This df should be with join keys.
    @param num_features: the number of features in the dataframe.
    @param key_domains: a dictionary containing the domain of each join key.
    @param join_keys: a list containing the names of the join keys.
    @param fit_by_residual: a boolean indicating whether to fit by residual.
    @param is_buyer: a boolean indicating whether the df is a buyer or seller.

    @return: a tuple containing the calibrated 1, x, x_x tensors
    """

    def _calibrate(self, df_id, df, num_features, key_domains, join_keys, normalized=True, fit_by_residual=False, is_buyer=False):
        # Get a squared df but not include join keys
        non_join_key_columns = df.columns.difference(join_keys)
        df_squared = df[non_join_key_columns] ** 2
        df_squared[join_keys] = df[join_keys]

        seller_sum = df.groupby(join_keys).sum()
        ordered_columns = list(seller_sum.columns)

        if df_id not in self.dfid_feature_mapping:
            self.dfid_feature_mapping[df_id] = ordered_columns
        else:
            self.dfid_feature_mapping[df_id] += ordered_columns
        
        seller_sum_squares = df_squared.groupby(join_keys).sum()[ordered_columns]
        seller_count = df.groupby(join_keys).size().to_frame('count')

        
        if not fit_by_residual and is_buyer:
            df_cross, ordered_cross_cols = {}, []
            for col1, col2 in combinations(ordered_columns, 2):
                df_cross[f"{col1}_{col2}"] = df[col1] * df[col2]
                ordered_cross_cols.append(f"{col1}_{col2}")
            df_cross = pd.DataFrame(df_cross)
            df_cross[join_keys] = df[join_keys]
            seller_sum_cross = df_cross.groupby(join_keys).sum()[ordered_cross_cols]
            if normalized:
                seller_sum_cross = seller_sum_cross.div(seller_count['count'], axis=0)
        
                
        # Normalize by seller_count if normalization is enabled
        if normalized:
            seller_sum = seller_sum.div(seller_count['count'], axis=0)
            seller_sum_squares = seller_sum_squares.div(seller_count['count'], axis=0)

            # Set seller_count to 1 for each group
            seller_count = seller_count.assign(count=1)

        if not isinstance(seller_sum.index, pd.MultiIndex):
            seller_sum.index = pd.MultiIndex.from_arrays(
                [seller_sum.index], names=join_keys)
            seller_sum_squares.index = pd.MultiIndex.from_arrays(
                [seller_sum_squares.index], names=join_keys)
            seller_count.index = pd.MultiIndex.from_arrays(
                [seller_count.index], names=join_keys)
            
            if not fit_by_residual and is_buyer:
                seller_sum_cross.index = pd.MultiIndex.from_arrays(
                    [seller_sum_cross.index], names=join_keys)
            
        # Create the correct multi_index for cartesian product
        index_ranges = [key_domains[col] for col in join_keys]
        multi_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)
        
        # Temporary DataFrame to facilitate 'inner' join
        temp_df = pd.DataFrame(index=multi_index)

        # Reindex and perform inner join
        seller_x = seller_sum.reindex(multi_index, fill_value=0)
        seller_x = seller_x[seller_x.index.isin(temp_df.index)].values

        seller_x_x = seller_sum_squares.reindex(multi_index, fill_value=0)
        seller_x_x = seller_x_x[seller_x_x.index.isin(temp_df.index)].values

        seller_count = seller_count.reindex(multi_index, fill_value=1)
        seller_count = seller_count[seller_count.index.isin(temp_df.index)].values
        
        seller_x_y_tensor = None
        
        if not fit_by_residual and is_buyer:
            seller_x_y = seller_sum_cross.reindex(multi_index, fill_value=0)
            seller_x_y = seller_x_y[seller_x_y.index.isin(temp_df.index)].values
            seller_x_y_tensor = torch.tensor(seller_x_y, dtype=torch.float32)
        
        # Convert to PyTorch tensors
        seller_x_tensor = torch.tensor(seller_x, dtype=torch.float32)
        seller_x_x_tensor = torch.tensor(seller_x_x, dtype=torch.float32)
        seller_count_tensor = torch.tensor(
            seller_count, dtype=torch.int).view(-1, 1)
        seller_1_tensor = seller_count_tensor.expand(-1, num_features)

        return seller_x_tensor, seller_x_x_tensor, seller_1_tensor, seller_x_y_tensor
    
    """
    This function gets the batch_id and the feature_index in this batch. These are all found in the searchEngine class.
    @param batch_id: the batch_id of the feature_index
    @param feature_index: the feature_index of the feature in the batch

    @return: a tuple containing the df_id and a feature name indicated by the dfid_feature_mapping
    """
    def get_df_by_feature_index(self, batch_id, feature_index):
        # Perform a binary search to find the right interval
        # bisect.bisect returns the insertion point which gives us the index where the feature_index would be inserted to maintain order.
        # We subtract one to get the tuple corresponding to the start index of the range that the feature_index falls into.
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:  # Compare with the first element of the tuple at mid
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        index = bisect(self.feature_index_mapping[batch_id], feature_index) - 1
        start_index, df_id, offset = self.feature_index_mapping[batch_id][index]
        # Calculate the local feature index within the seller's dataset
        local_feature_index = feature_index - start_index + offset
        return df_id, self.dfid_feature_mapping[df_id][local_feature_index]
    
    def get_sketch_loader(self):
        return self.sketch_loader
    
"""
This class wraps up a seller df to a sketch. It will be used to register the seller df to the sketch base 
and store the corresponding batch_id and offset. It also stores the related information such as join keys,
join key domains, and the sketch base object.
"""
class SellerSketch():
    def __init__(self, seller_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, df_id: int, device='cpu'):
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.all_join_keys = sorted(self.join_key_domains.keys())
        self.device = device
        self.df_id = df_id


        # Seller's dataframe will be stored in this variable
        self.seller_df = seller_df

        # This is from the sketch base. Will be updated after registering the seller sketch
        self.batch_id = 0
        self.offset = 0

        # This stores a seller sketch base. This single seller df will use the sketch base to register itself
        self.sketch_base = sketch_base


    """
    This function is used to register a seller df to the sketch base.

    @return: a tuple containing the batch_id and offset
    """
    def register_this_seller(self):
        # First we should cut the df into partitions to maximize the GPU and RAM usage
        ram_batch_size = self.sketch_base.ram_batch_size
        # Rename the columns and add the join keys as prefix to the column names
        prefix = "_".join(self.join_keys) + "_"
        self.seller_df.columns = [prefix + col if col not in self.join_keys else col for col in self.seller_df.columns]
        feature_columns = [col for col in self.seller_df.columns if col not in self.join_keys]
        if len(self.seller_df.columns) > ram_batch_size:
            features_per_partition = ram_batch_size - 1
            # Splitting the DataFrame into partitions
            num_partitions = (len(feature_columns) // features_per_partition) + (len(feature_columns) % features_per_partition > 0)
            for i in range(num_partitions):
                cur_features = feature_columns[i * features_per_partition:(i + 1) * features_per_partition] # Avoid address coding
                cols = self.join_keys + cur_features
                # Creating a new DataFrame for this partition
                cur_df = self.seller_df[cols]
                # Calibrate the df
                seller_x, seller_x_x, seller_1, seller_x_y = self.sketch_base._calibrate(
                    self.df_id, cur_df, len(cur_features), self.join_key_domains, self.join_keys)
                # Register the df
                result = self.sketch_base._register_df(self.df_id, len(cur_features), seller_1, seller_x, seller_x_x)
                self.batch_id = result["batch_id"]
                self.offset = result["offset"]
        else:
            # Directly calibrate the df
            seller_x, seller_x_x, seller_1, seller_x_y = self.sketch_base._calibrate(
                self.df_id, self.seller_df, len(self.seller_df.columns) - len(self.join_keys), self.join_key_domains, self.join_keys)
            # Register the df
            result = self.sketch_base._register_df(self.df_id, len(self.seller_df.columns) - len(self.join_keys), seller_1, seller_x, seller_x_x)
            self.batch_id = result["batch_id"]
            self.offset = result["offset"]
            # We don't update df_id here because it is the id of the seller_df
        

        return self.batch_id, self.offset
    
    def get_base(self):
        return self.sketch_base
    
    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)
    
    def get_df(self):
        return self.seller_df

"""
This class wraps up a buyer df to a sketch. It will be used to register the buyer df to the sketch base
and store the corresponding batch_id and offset. It also stores the related information such as join keys,
join key domains, and the sketch base object. 

One more thing it stores is the target feature and the corresponding index of the target feature in the buyer df.
"""
class BuyerSketch():
    def __init__(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict, sketch_base: SketchBase, target_feature: str, device='cpu', fit_by_residual=False):
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.device = device
        self.df_id = 0

        # This is to indicate the target feature of the buyer
        self.target_feature = target_feature
        if not fit_by_residual:
            # When fitting by residual, the target feature is not in the buyer_df
            self.target_feature_index = buyer_df.columns.get_loc(target_feature)

        # Buyer's dataframe will be stored in this variable
        self.buyer_df = buyer_df

        # This is from the sketch base. Will be updated after registering the buyer sketch
        self.batch_id = 0 # Since for now the buyer dataset has little chance to exceed the batch size, 
                            # we set the batch_id to 0. Also since the buyer dataset is small, 
                            # we don't need to split it into partitions. So it is not a list.
        self.offset = 0

        # This stores a buyer sketch base. This single buyer df will use the sketch base to register itself
        self.sketch_base = sketch_base


    """
    This function is used to register a buyer df to the sketch base. 

    @return: a tuple containing the batch_id and offset
    """
    def register_this_buyer(self, fit_by_residual=False):
        # Calibrate the df
        buyer_x, buyer_x_x, buyer_1, buyer_x_y = self.sketch_base._calibrate(
            self.df_id, self.buyer_df, len(self.buyer_df.columns) - len(self.join_keys), self.join_key_domains, self.join_keys, is_buyer=True, fit_by_residual=fit_by_residual)
        # Register the df
        result = self.sketch_base._register_df(df_id= self.df_id, feature_num=len(self.buyer_df.columns) - len(self.join_keys), seller_1=buyer_1, seller_x=buyer_x, seller_x_x=buyer_x_x, seller_x_y=buyer_x_y)
        self.batch_id = result["batch_id"]
        self.offset = result["offset"]
        # We don't update df_id here because it is the id of the buyer_df

        return self.batch_id, self.offset
    
    def get_base(self):
        return self.sketch_base
    
    """
    This function gets the 1, X, X_X, and X_Y matrices of the buyer df, which is stored in the sketch base.

    @return: a tuple containing the 1, X, X_X, and X_Y matrices
    """
    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)
    
    """
    This function gets the feature index and name of the target feature in the buyer df.

    @return: a dictionary containing the index and name of the target feature
    """
    def get_target_feature(self):
        return {"index": self.target_feature_index, "name": self.target_feature}


"""
This class would be used as a overall register for both seller and buyer sketches. It helps to prepare the 
buyer sketch base and seller sketch base for the search engine.
"""
class DataMarket():
    def __init__(self, device='cpu'):
        # Storing and initializing the sketch bases
        self.seller_sketches = {}       # join_key: [
                                        #           {id, join_key, join_key_domain, seller_sketch},
                                        #           {id, join_key, join_key_domain, seller_sketch}
                                        #         ]

        self.buyer_sketches = {}        # join_key: {id, join_key, join_key_domain, buyer_sketch}
        self.buyer_dataset_for_residual = None
        # Storing the df_id for each seller and buyer
        self.seller_id = 0
        self.buyer_id = 0

        self.buyer_target_feature = ""
        self.buyer_join_keys = []

        # Storing the id-name pair for each seller and buyer
        self.seller_id_to_df_and_name = []
        self.buyer_id_to_df_and_name = []

        self.augplan_acc = []

        # Device
        self.device = device

    """
    This function is used to register a seller df to the data market. It will create a SellerSketch object based
    on the seller_df, join_keys, join_key_domains, and the sketch_base. It will also maintain and update the seller_id and 
    seller_sketches dictionary.

    For each key in the join_keys, we will make a new SellerSketch object. So each SellerSketch object will only have one join key.
    
    @param seller_df: the seller dataframe to be registered
    @param join_keys: the join keys of the seller dataframe
    @param join_key_domains: the domain of each join key

    @return: seller_id
    """
    def register_seller(self, seller_df: pd.DataFrame, seller_name: str, join_keys: list, join_key_domains: dict):
        # To avoid the case where the seller_df are containing some features that have the same name as the registered features, 
        # we need to add a prefix for all the features except the join keys
        prefix = seller_name + "_"
        seller_df.columns = [prefix + col if col not in [i[j] for i in join_keys for j in range(len(i))] else col for col in seller_df.columns]
        for join_key in join_keys:
            join_key = join_key[0]
            if join_key in self.seller_sketches:
                seller_sketch_base = self.seller_sketches[join_key]["sketch_base"]
            else:
                seller_sketch_base = SketchBase(join_key_domain=join_key_domains, device=self.device)
                # Create a new list for the new join key
                self.seller_sketches[join_key] = {}
                self.seller_sketches[join_key]["sketch_base"] = seller_sketch_base
            try:
                seller_df_with_the_key = seller_df[list(seller_df.columns.difference([i[j] for i in join_keys for j in range(len(i))])) + [join_key]]
            except KeyError as e:
                continue
            # Create a SellerSketch object
            seller_sketch = SellerSketch(
                seller_df_with_the_key, 
                [join_key], 
                join_key_domains, 
                seller_sketch_base,
                self.seller_id, 
                self.device
            )
            # Register the seller and store the seller sketch object with related information
            seller_sketch_info = {}
            seller_sketch_info["id"] = self.seller_id
            seller_sketch_info["name"] = seller_name
            seller_sketch_info["join_key"] = join_key
            seller_sketch_info["join_key_domain"] = join_key_domains
            seller_sketch_info["seller_sketch"] = seller_sketch
            self.seller_sketches[join_key][self.seller_id] = seller_sketch_info

            
            batch_id, offset = seller_sketch.register_this_seller()

        self.seller_id_to_df_and_name.append(
            {"name": seller_name,
             "dataframe": seller_df}
        )
        # Update the seller_id
        self.seller_id += 1

        return self.seller_id - 1

    """
    This function is used to register a buyer df to the data market. It will create a BuyerSketch object based
    on the buyer_df, join_keys, join_key_domains, and the sketch_base. It will also maintain and update the buyer_id and
    buyer_sketches dictionary. Additionally, it will take in the target_feature as an input parameter.

    For each key in the join_keys, we will make a new BuyerSketch object. So each BuyerSketch object will only have one join key.

    @param buyer_df: the buyer dataframe to be registered
    @param join_keys: the join keys of the buyer dataframe
    @param join_key_domains: the domain of each join key
    @param target_feature: the target feature of the buyer dataframe

    @return: buyer_id
    """
    def register_buyer(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict, target_feature: str, fit_by_residual=False):     
        if fit_by_residual:
            self.buyer_dataset_for_residual = copy.deepcopy(buyer_df)
        self.buyer_dataset = copy.deepcopy(buyer_df)
        self.buyer_join_keys = join_keys
        self.buyer_target_feature = target_feature
        X = list(self.buyer_dataset.columns.difference([target_feature] + join_keys))
        # Calculate the residuals from linear regression
        res, r2 = linear_regression_residuals(self.buyer_dataset, X_columns=X, Y_column=target_feature, adjusted=False)
        self.augplan_acc.append(r2)
        if fit_by_residual:
            self.buyer_dataset = res[join_keys + ["residuals"]]
        else:
            self.buyer_dataset = self.buyer_dataset.drop(columns=["residuals"], errors="ignore")

        for join_key in join_keys:
            if join_key in self.buyer_sketches:
                buyer_sketch_base = self.buyer_sketches[join_key]["buyer_sketch"].get_base()
            else:
                buyer_sketch_base = SketchBase(join_key_domain=join_key_domains, device=self.device, is_buyer=True)
            buyer_df_with_the_key = self.buyer_dataset[list(self.buyer_dataset.columns.difference(join_keys)) + [join_key]]
            # Create a BuyerSketch object
            buyer_sketch = BuyerSketch(
                buyer_df_with_the_key, 
                [join_key], 
                join_key_domains, 
                buyer_sketch_base, 
                target_feature, 
                self.device,
                fit_by_residual
            )
            # Register the buyer and store the buyer sketch object with related information
            buyer_sketch_info = {}
            buyer_sketch_info["id"] = self.buyer_id
            buyer_sketch_info["join_key"] = join_key
            buyer_sketch_info["join_key_domain"] = join_key_domains
            buyer_sketch_info["buyer_sketch"] = buyer_sketch
            self.buyer_sketches[join_key] = buyer_sketch_info

            batch_id, offset = buyer_sketch.register_this_buyer(fit_by_residual=fit_by_residual) # Currently, batch_id and offset are not used

        # Store buyer_dataset_for_residual which contains all features (not buyer_dataset which only has residuals)
        self.buyer_id_to_df_and_name.append(
            {"name": target_feature,
             "dataframe": self.buyer_dataset_for_residual if fit_by_residual else self.buyer_dataset}
        )
        # Update the buyer_id
        self.buyer_id += 1

        

        return self.buyer_id - 1

    
    """
    This function is used to get the seller sketch base.
    """
    def get_seller_sketch_base(self):
        return self.seller_sketch_base
    
    """
    This function is used to get the buyer sketch base.
    """
    def get_buyer_sketch_base(self):
        return self.buyer_sketch_base
    
    """
    This function is used to get the buyer sketch object based on the buyer_id.
    """
    def get_buyer_sketch(self, buyer_id):
        return self.buyer_sketches[buyer_id]["buyer_sketch"]
    
    """
    This function gets the seller sketch object based on the join_key.
    """
    def get_seller_sketch_by_keys(self, join_key, seller_id):
        return self.seller_sketches[join_key][seller_id]["seller_sketch"]
    
    """
    This function gets the seller sketch base based on the join_key.
    """
    def get_seller_sketch_base_by_keys(self, join_key):
        return self.seller_sketches[join_key]["sketch_base"]
    
    """
    This function gets the buyer sketch object based on the join_key.
    """
    def get_buyer_sketch_by_keys(self, join_key):
        return self.buyer_sketches[join_key]["buyer_sketch"]
    
    """
    This function sets the buyer_id
    """
    def set_buyer_id(self, buyer_id):
        self.buyer_id = buyer_id

    """
    This function resets the buyer sketches
    """
    def reset_buyer_sketches(self):
        self.buyer_sketches = {}

    """
    This function resets the buyer_id_to_df_and_name
    """
    def reset_buyer_id_to_df_and_name(self):
        self.buyer_id_to_df_and_name = []
    

class SearchEngine():

    def __init__(self, data_market: DataMarket, fit_by_residual=False):
        self.augplan = []
        self.augplan_acc = []
        self.aug_seller_feature_ind = {}
        self.buyer_target = data_market.buyer_target_feature
        self.buyer_features = data_market.buyer_dataset_for_residual.columns
        self.buyer_dataset = None  # placeholder for buyer dataset
        self.buyer_sketches = {}  # placeholder for buyer join sketches
        self.seller_sketches = {}  # placeholder for seller join sketches
        self.fit_by_residual = fit_by_residual
        self.data_market = data_market

        self.seller_aggregated = {} # for seller aggregation by join key

        self.unusable_features = {} # {batch_id: [feature_index]}

    """
    This function is used to search for the best seller for a buyer. It will iterate through all the join keys and
    find the best seller for each join key. It will return the best seller for each join key.

    @return: a dictionary containing the best seller for each join key
    """
    def search_one_iteration(self, best_r_squared):
        
        best_r_squared_ind = -1
        best_batch_id = -1
        best_join_key = None

        # Get buyer_sketches from data_market
        self.buyer_sketches = self.data_market.buyer_sketches # join_key: {id, join_key, join_key_domain, buyer_sketch}
        for join_key in sorted(self.buyer_sketches.keys()):
            buyer_id = self.buyer_sketches[join_key]["id"]
            buyer_join_key_domain = self.buyer_sketches[join_key]["join_key_domain"]
            buyer_sketch = self.buyer_sketches[join_key]["buyer_sketch"]

            # Get the buyer sketches
            buyer_1, buyer_y, buyer_y_y, buyer_x_y = buyer_sketch.get_sketches()

            # Get the search_sketch for the buyer (which is just the seller sketch with join_key)
            search_sketch_base = self.data_market.get_seller_sketch_base_by_keys(join_key)
            # print feature names of this sketch

            for batch_id in range(search_sketch_base.get_sketch_loader().get_num_batches()):
                seller_1, seller_x, seller_x_x, _ =search_sketch_base.get_sketch_loader().get_sketches(batch_id)
                if not self.fit_by_residual:
                    d = buyer_y.shape[1]
                    ordered_columns = buyer_sketch.get_base().dfid_feature_mapping[buyer_id]
                    y_ind = ordered_columns.index(buyer_sketch.get_target_feature()["name"])
                    
                    # Algortihm
                    XTX = torch.zeros(seller_x.shape[1], d+1, d+1).to(self.data_market.device)
                    XTY = torch.zeros(seller_x.shape[1], d+1, 1).to(self.data_market.device)
                    c = torch.sum(buyer_1 * seller_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x_x[x_x == 0] = 1
                    y = torch.sum(buyer_y[:, y_ind:y_ind+1] * seller_1, dim=0) 
                    y_y = torch.sum(buyer_y_y[:, y_ind:y_ind+1] * seller_1, dim=0)
                    TSS = y_y - y * y / c
                    XTX[:, 0, 0] = c
                    XTX[:, 0, 1] = XTX[:, 1, 0] = x
                    XTX[:, 1, 1] = x_x
                    for i in range(d):
                        cur_buyer_y = buyer_y[:, i:i+1]
                        cur_buyer_y_y = buyer_y_y[:, i:i+1]

                        cur_x_y = torch.sum(seller_x * cur_buyer_y, dim=0)
                        cur_y_y = torch.sum(cur_buyer_y_y * seller_1, dim=0)
                        cur_y = torch.sum(cur_buyer_y * seller_1, dim=0)
                        cur_y_y[cur_y_y == 0] = 1

                        if i == y_ind:
                            XTY[:, 0, 0] = cur_y
                            XTY[:, 1, 0] = cur_x_y
                        elif i < y_ind:
                            XTX[:, i+2, i+2] = cur_y_y
                            XTX[:, 1, i+2] = XTX[:, i+2, 1] = cur_x_y
                            XTX[:, 0, i+2] = XTX[:, i+2, 0] = cur_y
                        else:
                            XTX[:, i+1, i+1] = cur_y_y
                            XTX[:, 1, i+1] = XTX[:, i+1, 1] = cur_x_y
                            XTX[:, 0, i+1] = XTX[:, i+1, 0] = cur_y
                        
                        # (2d-i)(i-1)/2 +j-i's column
                        for j in range(i+1, d):
                            x_y_ind = int((2*d-i-1)*i/2+j-i)-1
                            x_y_ = torch.sum(buyer_x_y[:, x_y_ind:x_y_ind+1] * seller_1, dim=0)
                            if i == y_ind:
                                XTY[:, j+1, 0] = x_y_
                            elif j == y_ind:
                                XTY[:, i+2, 0] = x_y_
                            elif i > y_ind:
                                XTX[:, i+1, j+1] = XTX[:, j+1, i+1] = x_y_
                            elif i < y_ind and j > y_ind:
                                XTX[:, i+2, j+1] = XTX[:, j+1, i+2] = x_y_
                            else:
                                XTX[:, i+2, j+2] = XTX[:, j+2, i+2] = x_y_

                    inverses = torch.empty_like(XTX)

                    for i in range(len(XTX)):
                        try:
                            # Try to calculate the inverse of the matrix
                            inverses[i] = torch.linalg.inv(XTX[i])
                        except RuntimeError as e:
                            # If the matrix is singular, store the feature index and batch id
                            print(f"[Warning] Singular matrix at batch {batch_id} and feature index {i}, corresponding to feature { self.data_market.get_seller_sketch_base_by_keys(join_key).get_df_by_feature_index(batch_id, i)[1]}")
                            self.unusable_features[batch_id] = self.unusable_features.get(batch_id, [])
                            self.unusable_features[batch_id].append(i)
                            # Set this layer to a zero matrix
                            inverses[i] = torch.zeros_like(XTX[i])

                    res = torch.bmm(inverses, XTY).to(self.data_market.device)
                    RSS = y_y
                    for i in range(d+1):
                        for j in range(d+1):
                            RSS += res[:, i, 0]*res[:, j, 0]*XTX[:, i, j]
                            
                        RSS -= 2*res[:, i, 0]*XTY[:, i, 0]
                    r_squared = 1 - RSS / TSS
                else:
                    # TODO: might be singular, how to deal with it
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    c = torch.sum(buyer_1 * seller_1, dim=0)

                    x_y = torch.sum(seller_x * buyer_y, dim=0)
                    y_y = torch.sum(buyer_y_y * seller_1, dim=0)
                    y = torch.sum(buyer_y * seller_1, dim=0)

                    # Calculate means
                    x_mean = x / c
                    y_mean = y / c

                    # Calculate the components needed for the formulas
                    # Sum of squares of deviations is often denoted as S_xx (for x) and S_yy (for y)
                    # and sum of cross-deviations as S_xy
                    S_xx = x_x - 2 * x_mean * x + c * x_mean ** 2
                    S_xy = x_y - x_mean * y - x * y_mean + c * x_mean * y_mean

                    # Calculate regression coefficients, i.e., slope and intercept for each set
                    # Formula for the slope (beta or m): S_xy / S_xx
                    # Formula for the intercept (alpha or c): y_mean - m * x_mean
                    slope = S_xy / S_xx
                    intercept = y_mean - slope * x_mean

                    TSS = y_y - 2 * y_mean * y + c * y_mean ** 2
                    RSS = y_y + c * intercept ** 2 + slope ** 2 * x_x - 2 * \
                        (slope * x_y + intercept * y - slope * intercept * x)

                    # Calculate R² value for each regression model
                    # R² = 1 - (RSS / TSS)
                    r_squared = 1 - (RSS / TSS)
                # Replace NaN values with negative infinity
                r_squared = torch.where(torch.isnan(r_squared), torch.tensor(float('-inf')), r_squared)
                r_squared = torch.where(r_squared >= 1, torch.tensor(float('-inf')), r_squared)

                # Replace singular feature indices with negative infinity
                if batch_id in self.unusable_features:
                    for singular_ind in self.unusable_features[batch_id]:
                        r_squared[singular_ind] = float('-inf')

                
                if join_key in self.aug_seller_feature_ind and batch_id in self.aug_seller_feature_ind[join_key]:
                    exclude_indices = self.aug_seller_feature_ind[join_key][batch_id]
                    # Temporarily set the values at the excluded indices to negative infinity
                    original_values = r_squared[exclude_indices].clone()
                    r_squared[exclude_indices] = float('-inf')
                    # Find the index of the maximum value
                    max_r2_index = torch.argmax(r_squared)

                    if r_squared[max_r2_index].item() < -1:
                        # This means that all the features have been selected
                        r_squared[exclude_indices] = original_values
                        continue

                    # Restore the original values at the excluded indices
                    r_squared[exclude_indices] = original_values
                else:
                    # Find the index of the maximum value
                    try:
                        max_r2_index = torch.argmax(r_squared)
                    except IndexError:
                        return None, None, None, None
                # max_r2_index = torch.argmax(r_squared)
                if r_squared[max_r2_index].item() > best_r_squared:
                    best_r_squared = r_squared[max_r2_index].item()
                    best_r_squared_ind = max_r2_index
                    best_batch_id = batch_id
                    best_join_key = join_key
                if not self.fit_by_residual:
                    cleanup(x_x, x, c, y, y_y, inverses, res, TSS,
                            RSS, r_squared, seller_1, seller_x, seller_x_x)
                else:
                    cleanup(x_x, x, c, y, y_y, x_y, x_mean, y_mean, S_xx, S_xy, TSS,
                            RSS, r_squared, slope, intercept, seller_1, seller_x, seller_x_x)
                    
        if best_r_squared_ind == -1:
            return None, None, None, None
        else: 
            print("Maximum R² value:", best_r_squared)
            
            return best_join_key, best_r_squared_ind.item(), best_batch_id, best_r_squared
        
    """
    This function is used to start the search engine. It will iterate through the search_one_iteration function
    for a certain number of iterations. It will store the best features for each seller and update the buyer sketch
    based on the new features.
    """
    def start(self, iter=2):
        for i in range(iter):
            # Each iteration starts fresh - find the best feature among remaining features
            join_key, ind, batch_id, best_r_squared = self.search_one_iteration(best_r_squared=-1.0)
            if not join_key:
                print("No more good features")
                break
            if join_key not in self.aug_seller_feature_ind:
                self.aug_seller_feature_ind[join_key] = {
                    batch_id: torch.tensor([ind])}
            elif batch_id not in self.aug_seller_feature_ind[join_key]:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.tensor([
                                                                               ind])
            else:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.cat(
                    (self.aug_seller_feature_ind[join_key][batch_id], torch.tensor([ind])))
            seller_id, best_feature = self.data_market.get_seller_sketch_base_by_keys(join_key).get_df_by_feature_index(
                batch_id, ind)
            print("The best feature in iter ", i, " is:", best_feature, "with join key ", join_key)
            
            # Store the current overall model R² before updating
            # Get from the augplan list we're maintaining in SearchEngine, not DataMarket
            if i == 0:
                # First iteration: get baseline from DataMarket (set during initial register_buyer)
                prev_overall_r2 = self.data_market.augplan_acc[0] if self.data_market.augplan_acc else 0.0
            else:
                # Later iterations: get from our own tracking list
                prev_overall_r2 = self.augplan_acc[-1]
            
            # Tentatively update residual to check if R² improves
            self._update_residual(join_key, seller_id, best_feature)
            
            # After update, check if overall model R² improved
            new_overall_r2 = self.data_market.augplan_acc[-1]
            # Calculate percentage-wise improvement
            if prev_overall_r2 != 0:
                improvement_pct = (new_overall_r2 - prev_overall_r2) / abs(prev_overall_r2) * 100
            else:
                improvement_pct = float('inf') if new_overall_r2 > 0 else 0.0
            print(f"Overall model R² after adding feature: {new_overall_r2:.6f} (previous: {prev_overall_r2:.6f}, improvement: {improvement_pct:.2f}%)")
            
            # Only add to augmentation plan if R² improved by at least 1%
            min_improvement_pct = 1.0
            if improvement_pct >= min_improvement_pct:
                self.augplan.append((seller_id, 
                                     i+1, 
                                     self.data_market.seller_id_to_df_and_name[seller_id]["name"],
                                     best_feature))
                self.augplan_acc.append(new_overall_r2)
                print(f"✓ Feature added (improvement: {improvement_pct:.2f}% >= {min_improvement_pct}%)")
            else:
                print(f"✗ Overall model R² did not improve enough (improvement: {improvement_pct:.2f}% < {min_improvement_pct}%). Stopping feature selection.")
                break
        return self.augplan, self.augplan_acc, self.data_market.buyer_dataset if not self.fit_by_residual else self.data_market.buyer_dataset_for_residual
            
    def _update_residual(self, join_key, seller_id, best_feature):
        buyer = self.data_market.buyer_id_to_df_and_name[0]["dataframe"]
        if self.fit_by_residual:
            buyer = self.data_market.buyer_dataset_for_residual
        seller_df = self.data_market.get_seller_sketch_by_keys(join_key=join_key, seller_id=seller_id).get_df()[[join_key, best_feature]]

        # Pre-aggregate seller_df by join keys
        
        aggregation_functions = {col: 'mean' for col in seller_df.columns if col != join_key}
        seller_df_agg = seller_df.groupby(join_key).agg(aggregation_functions).reset_index()

        # Normalize join keys to prevent dtype mismatch errors
        buyer[join_key] = buyer[join_key].apply(process_key)
        seller_df_agg[join_key] = seller_df_agg[join_key].apply(process_key)
        
        # left join buyer_df with seller_df_agg on join key
        joined_df = pd.merge(buyer, seller_df_agg, how='left', left_on=join_key, right_on=join_key, suffixes=('_left', '_right'))
        # Remove the _right columns and rename the _left columns to the original column names
        joined_df = joined_df[[col for col in joined_df.columns if '_right' not in col]]
        joined_df.columns = [col.replace('_left', '') for col in joined_df.columns]
        
        # Fill in nulls in the new features with the mean of their columns
        for col in seller_df_agg.columns:
            if col != join_key:
                joined_df[col].fillna(joined_df[col].mean(), inplace=True)

        updated_buyer = joined_df[list(set([join_key] + list(buyer.columns) + [best_feature] + [self.buyer_target]))]

        # Update the buyer sketch
        buy_keys = self.data_market.buyer_join_keys
        # Get the full join_key_domains dict (not just for one key)
        join_key_domains = self.data_market.buyer_sketches[join_key]["join_key_domain"]

        self.data_market.set_buyer_id(0)
        self.data_market.reset_buyer_sketches()
        self.data_market.reset_buyer_id_to_df_and_name()
        self.data_market.register_buyer(updated_buyer, buy_keys,
                                        join_key_domains, self.buyer_target, fit_by_residual=self.fit_by_residual)
