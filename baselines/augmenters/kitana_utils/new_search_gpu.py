"""Sketch-based GPU search for Kitana.

Mirrors the structure of cudbg/Kitana-e2e (search_engine/{sketches,market,search})
but flattened into a single module to preserve this codebase's existing import
surface (``DataMarket``, ``SearchEngine``).

User-facing carve-outs vs. upstream:
- Device, batch size, and ``fit_by_residual`` are passed through constructors
  (upstream pulls these from a global Config singleton that does not exist here).
- ``DataMarket.register_seller`` accepts the legacy list-of-list ``join_keys``
  shape produced by ``PrepareSeller.join_keys``.
- ``SearchEngine._update_residual`` normalises join keys via ``process_key``
  before the buyer/seller merge to defend against dtype drift in lake CSVs.
"""

import os
import copy
import bisect
import psutil
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from matryoshka.utils.common import process_key


def cleanup(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            del arg
    torch.cuda.empty_cache()


def linear_regression_residuals(df, X_columns, Y_column, adjusted=False):
    if not all(item in df.columns for item in X_columns):
        raise ValueError('Not all specified X_columns are in the dataframe.')
    if Y_column not in df.columns:
        raise ValueError('The Y_column is not in the dataframe.')

    X = df[X_columns].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    Y = df[Y_column].values

    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    residuals = Y - Y_pred
    df['residuals'] = residuals

    SS_res = (residuals ** 2).sum()
    SS_tot = ((Y - np.mean(Y)) ** 2).sum()
    R_squared = 1 - SS_res / SS_tot

    if adjusted:
        n = X.shape[0]
        p = X.shape[1] - 1
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
            self.sketch_1_batch[0] = seller_1[:, :min(self.batch_size, seller_1.size(1))]
            remaining_seller_1 = seller_1[:, self.batch_size:]
            self.sketch_x_batch[0] = seller_x[:, :min(self.batch_size, seller_x.size(1))]
            remaining_seller_x = seller_x[:, self.batch_size:]
            self.sketch_x_x_batch[0] = seller_x_x[:, :min(self.batch_size, seller_x_x.size(1))]
            remaining_seller_x_x = seller_x_x[:, self.batch_size:]
            feature_index_map[0] = [(0, seller_id, 0)]
            cur_df_offset = self.batch_size
        else:
            last_batch_num = max(self.sketch_x_batch.keys())
            last_batch_1 = self.sketch_1_batch[last_batch_num]
            last_batch_x = self.sketch_x_batch[last_batch_num]
            last_batch_x_x = self.sketch_x_x_batch[last_batch_num]

            remaining_space = self.batch_size - last_batch_x.size(1)

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
                bisect.insort(feature_index_map[last_batch_num],
                              (last_batch_x.size(1), seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                last_batch_num += 1
                self.sketch_1_batch[last_batch_num] = seller_1[:, :min(self.batch_size, seller_1.size(1))]
                self.sketch_x_batch[last_batch_num] = seller_x[:, :min(self.batch_size, seller_x.size(1))]
                self.sketch_x_x_batch[last_batch_num] = seller_x_x[:, :min(self.batch_size, seller_x_x.size(1))]
                remaining_seller_1 = seller_1[:, self.batch_size:]
                remaining_seller_x = seller_x[:, self.batch_size:]
                remaining_seller_x_x = seller_x_x[:, self.batch_size:]
                feature_index_map[last_batch_num] = [(0, seller_id, cur_df_offset)]
                cur_df_offset += self.batch_size
        self.num_batches = len(self.sketch_x_batch.keys())

        if remaining_seller_x.size(1) > 0:
            if not os.path.exists(self.disk_dir):
                os.makedirs(self.disk_dir)
            if to_disk:
                prev_batch_id = self.num_batches - 1
                torch.save(self.sketch_1_batch[prev_batch_id],
                           os.path.join(self.disk_dir, f"sketch_1_{prev_batch_id}.pt"))
                torch.save(self.sketch_x_batch[prev_batch_id],
                           os.path.join(self.disk_dir, f"sketch_x_{prev_batch_id}.pt"))
                torch.save(self.sketch_x_x_batch[prev_batch_id],
                           os.path.join(self.disk_dir, f"sketch_x_x_{prev_batch_id}.pt"))
                del self.sketch_1_batch[prev_batch_id]
                del self.sketch_x_batch[prev_batch_id]
                del self.sketch_x_x_batch[prev_batch_id]
            self.load_sketches(remaining_seller_1, remaining_seller_x, remaining_seller_x_x,
                               feature_index_map, seller_id, cur_df_offset, to_disk)

    def get_sketches(self, batch_id, from_disk=False):
        sketch_x_y_batch = None
        if from_disk:
            sketch_1_batch = torch.load(os.path.join(self.disk_dir, f"sketch_1_{batch_id}.pt")).to(self.device)
            sketch_x_batch = torch.load(os.path.join(self.disk_dir, f"sketch_x_{batch_id}.pt")).to(self.device)
            sketch_x_x_batch = torch.load(os.path.join(self.disk_dir, f"sketch_x_x_{batch_id}.pt")).to(self.device)
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
        self.sketch_loader = SketchLoader(self.gpu_batch_size, device=device, is_buyer=is_buyer)

    def estimate_batch_size(self):
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 3 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 3 * tensor_width)
        return gpu_batch_size, ram_batch_size

    def _register_df(self, df_id, feature_num, seller_1, seller_x, seller_x_x,
                     seller_x_y=None, to_disk=False):
        self.sketch_loader.load_sketches(
            seller_1=seller_1,
            seller_x=seller_x,
            seller_x_x=seller_x_x,
            seller_x_y=seller_x_y,
            feature_index_map=self.feature_index_mapping,
            seller_id=df_id,
            to_disk=to_disk,
        )

        def find_by_seller_id(feature_index_map, seller_id):
            for batch_id, entries in feature_index_map.items():
                for _end_pos, id_, offset in entries:
                    if id_ == seller_id:
                        return batch_id, offset
            return None, None

        batch_id, offset = find_by_seller_id(self.feature_index_mapping, df_id)
        return {"batch_id": batch_id, "df_id": df_id, "offset": offset}

    def _calibrate(self, df_id, df, num_features, key_domains, join_keys,
                   normalized=True, fit_by_residual=False, is_buyer=False):
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

        if normalized:
            seller_sum = seller_sum.div(seller_count['count'], axis=0)
            seller_sum_squares = seller_sum_squares.div(seller_count['count'], axis=0)
            seller_count = seller_count.assign(count=1)

        if not isinstance(seller_sum.index, pd.MultiIndex):
            seller_sum.index = pd.MultiIndex.from_arrays([seller_sum.index], names=join_keys)
            seller_sum_squares.index = pd.MultiIndex.from_arrays([seller_sum_squares.index], names=join_keys)
            seller_count.index = pd.MultiIndex.from_arrays([seller_count.index], names=join_keys)
            if not fit_by_residual and is_buyer:
                seller_sum_cross.index = pd.MultiIndex.from_arrays([seller_sum_cross.index], names=join_keys)

        index_ranges = [key_domains[col] for col in join_keys]
        multi_index = pd.MultiIndex.from_product(index_ranges, names=join_keys)
        temp_df = pd.DataFrame(index=multi_index)

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

        seller_x_tensor = torch.tensor(seller_x, dtype=torch.float32)
        seller_x_x_tensor = torch.tensor(seller_x_x, dtype=torch.float32)
        seller_count_tensor = torch.tensor(seller_count, dtype=torch.int).view(-1, 1)
        seller_1_tensor = seller_count_tensor.expand(-1, num_features)

        return seller_x_tensor, seller_x_x_tensor, seller_1_tensor, seller_x_y_tensor

    def get_df_by_feature_index(self, batch_id, feature_index):
        def bisect_(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        index = bisect_(self.feature_index_mapping[batch_id], feature_index) - 1
        start_index, df_id, offset = self.feature_index_mapping[batch_id][index]
        local_feature_index = feature_index - start_index + offset
        return df_id, self.dfid_feature_mapping[df_id][local_feature_index]

    def get_sketch_loader(self):
        return self.sketch_loader


class SellerSketch:
    def __init__(self, seller_df: pd.DataFrame, join_keys: list, join_key_domains: dict,
                 sketch_base: SketchBase, df_id: int, device='cpu'):
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.all_join_keys = sorted(self.join_key_domains.keys())
        self.device = device
        self.df_id = df_id
        # Drop non-numeric feature columns. _calibrate sums them via groupby
        # and the downstream torch.tensor(..., dtype=torch.float32) cast can't
        # consume string values (e.g. an unencoded DBN-typed column).
        join_key_set = set(join_keys)
        keep = [c for c in seller_df.columns
                if c in join_key_set or pd.api.types.is_numeric_dtype(seller_df[c])]
        if len(keep) < len(seller_df.columns):
            seller_df = seller_df[keep].copy()
        self.seller_df = seller_df
        self.batch_id = 0
        self.offset = 0
        self.sketch_base = sketch_base

    def register_this_seller(self):
        ram_batch_size = self.sketch_base.ram_batch_size
        prefix = "_".join(self.join_keys) + "_"
        self.seller_df.columns = [prefix + col if col not in self.join_keys else col
                                  for col in self.seller_df.columns]
        feature_columns = [col for col in self.seller_df.columns if col not in self.join_keys]
        if len(self.seller_df.columns) > ram_batch_size:
            features_per_partition = ram_batch_size - 1
            num_partitions = (len(feature_columns) // features_per_partition) + (
                len(feature_columns) % features_per_partition > 0)
            for i in range(num_partitions):
                cur_features = feature_columns[i * features_per_partition:(i + 1) * features_per_partition]
                cols = self.join_keys + cur_features
                cur_df = self.seller_df[cols]
                seller_x, seller_x_x, seller_1, _ = self.sketch_base._calibrate(
                    self.df_id, cur_df, len(cur_features), self.join_key_domains, self.join_keys)
                result = self.sketch_base._register_df(
                    self.df_id, len(cur_features), seller_1, seller_x, seller_x_x)
                self.batch_id = result["batch_id"]
                self.offset = result["offset"]
        else:
            seller_x, seller_x_x, seller_1, _ = self.sketch_base._calibrate(
                self.df_id, self.seller_df,
                len(self.seller_df.columns) - len(self.join_keys),
                self.join_key_domains, self.join_keys)
            result = self.sketch_base._register_df(
                self.df_id, len(self.seller_df.columns) - len(self.join_keys),
                seller_1, seller_x, seller_x_x)
            self.batch_id = result["batch_id"]
            self.offset = result["offset"]
        return self.batch_id, self.offset

    def get_base(self):
        return self.sketch_base

    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)

    def get_df(self):
        return self.seller_df


class BuyerSketch:
    def __init__(self, buyer_df: pd.DataFrame, join_keys: list, join_key_domains: dict,
                 sketch_base: SketchBase, target_feature: str, device='cpu', fit_by_residual=False):
        self.join_keys = join_keys
        self.join_key_domains = join_key_domains
        self.device = device
        self.df_id = 0
        self.target_feature = target_feature
        # Drop non-numeric feature columns before computing target_feature_index;
        # the join keys and the target are kept regardless of dtype.
        join_key_set = set(join_keys)
        keep = [c for c in buyer_df.columns
                if c in join_key_set or c == target_feature
                or pd.api.types.is_numeric_dtype(buyer_df[c])]
        if len(keep) < len(buyer_df.columns):
            buyer_df = buyer_df[keep].copy()
        if not fit_by_residual:
            self.target_feature_index = buyer_df.columns.get_loc(target_feature)
        self.buyer_df = buyer_df
        self.batch_id = 0
        self.offset = 0
        self.sketch_base = sketch_base

    def register_this_buyer(self, fit_by_residual=False):
        buyer_x, buyer_x_x, buyer_1, buyer_x_y = self.sketch_base._calibrate(
            self.df_id, self.buyer_df,
            len(self.buyer_df.columns) - len(self.join_keys),
            self.join_key_domains, self.join_keys,
            is_buyer=True, fit_by_residual=fit_by_residual)
        result = self.sketch_base._register_df(
            df_id=self.df_id,
            feature_num=len(self.buyer_df.columns) - len(self.join_keys),
            seller_1=buyer_1, seller_x=buyer_x, seller_x_x=buyer_x_x,
            seller_x_y=buyer_x_y)
        self.batch_id = result["batch_id"]
        self.offset = result["offset"]
        return self.batch_id, self.offset

    def get_base(self):
        return self.sketch_base

    def get_sketches(self):
        return self.sketch_base.sketch_loader.get_sketches(self.batch_id)

    def get_target_feature(self):
        return {"index": self.target_feature_index, "name": self.target_feature}


def _flatten_join_keys(join_keys):
    """Accept either ``[['k1'], ['k2']]`` or ``['k1', 'k2']`` and yield strings."""
    flat = []
    for jk in join_keys:
        if isinstance(jk, (list, tuple)):
            flat.extend(jk)
        else:
            flat.append(jk)
    return flat


class DataMarket:
    def __init__(self, device='cpu'):
        self.seller_sketches = {}
        self.buyer_sketches = {}
        self.buyer_dataset = None
        self.buyer_dataset_for_residual = None
        self.seller_id = 0
        self.buyer_id = 0
        self.buyer_target_feature = ""
        self.buyer_join_keys = []
        self.seller_id_to_df_and_name = []
        self.buyer_id_to_df_and_name = []
        self.augplan_acc = []
        self.device = device

    def register_seller(self, seller_df: pd.DataFrame, seller_name: str,
                        join_keys: list, join_key_domains: dict):
        flat_join_keys = _flatten_join_keys(join_keys)
        # Normalise join-key data and domains via process_key so seller and
        # buyer use identical preprocessing (lowercase, whitespace-stripped,
        # ASCII-only). Mismatched casing previously caused every seller to be
        # dropped at the buyer-seller intersection check.
        seller_df = seller_df.copy()
        for jk in flat_join_keys:
            if jk in seller_df.columns:
                seller_df[jk] = seller_df[jk].apply(process_key)
        join_key_domains = {
            k: ([process_key(v) for v in vs] if k in flat_join_keys else vs)
            for k, vs in join_key_domains.items()
        }
        prefix = seller_name + "_"
        seller_df.columns = [prefix + col if col not in flat_join_keys else col
                             for col in seller_df.columns]
        for join_key in flat_join_keys:
            if join_key in self.seller_sketches:
                seller_sketch_base = self.seller_sketches[join_key]["sketch_base"]
            else:
                seller_sketch_base = SketchBase(
                    join_key_domain=join_key_domains, device=self.device)
                self.seller_sketches[join_key] = {"sketch_base": seller_sketch_base}
            try:
                seller_df_with_the_key = seller_df[
                    list(seller_df.columns.difference(flat_join_keys)) + [join_key]]
            except KeyError:
                continue
            seller_sketch = SellerSketch(
                seller_df_with_the_key,
                [join_key],
                join_key_domains,
                seller_sketch_base,
                self.seller_id,
                self.device,
            )
            self.seller_sketches[join_key][self.seller_id] = {
                "id": self.seller_id,
                "name": seller_name,
                "join_key": join_key,
                "join_key_domain": join_key_domains,
                "seller_sketch": seller_sketch,
            }
            seller_sketch.register_this_seller()

        self.seller_id_to_df_and_name.append(
            {"name": seller_name, "dataframe": seller_df})
        self.seller_id += 1
        return self.seller_id - 1

    def register_buyer(self, buyer_df: pd.DataFrame, join_keys: list,
                       join_key_domains: dict, target_feature: str, fit_by_residual=False):
        # Mirror the normalisation applied in ``register_seller`` so the
        # buyer-seller intersection check sees identically processed keys.
        flat_join_keys = _flatten_join_keys(join_keys)
        buyer_df = buyer_df.copy()
        for jk in flat_join_keys:
            if jk in buyer_df.columns:
                buyer_df[jk] = buyer_df[jk].apply(process_key)
        join_key_domains = {
            k: ([process_key(v) for v in vs] if k in flat_join_keys else vs)
            for k, vs in join_key_domains.items()
        }
        if fit_by_residual:
            self.buyer_dataset_for_residual = copy.deepcopy(buyer_df)
        self.buyer_dataset = copy.deepcopy(buyer_df)
        self.buyer_join_keys = join_keys
        self.buyer_target_feature = target_feature
        # Use only numeric non-join, non-target columns as regressors;
        # LinearRegression.fit chokes on string columns (e.g. raw DBN-typed
        # categorical school columns).
        candidate_X = list(self.buyer_dataset.columns.difference([target_feature] + join_keys))
        X = [c for c in candidate_X
             if pd.api.types.is_numeric_dtype(self.buyer_dataset[c])]
        res, r2 = linear_regression_residuals(
            self.buyer_dataset, X_columns=X, Y_column=target_feature, adjusted=True)
        self.augplan_acc.append(r2)
        if fit_by_residual:
            self.buyer_dataset = res[join_keys + ["residuals"]]
        else:
            self.buyer_dataset = self.buyer_dataset.drop(columns=["residuals"], errors="ignore")

        for join_key in join_keys:
            if join_key in self.buyer_sketches:
                buyer_sketch_base = self.buyer_sketches[join_key]["buyer_sketch"].get_base()
            else:
                buyer_sketch_base = SketchBase(
                    join_key_domain=join_key_domains, device=self.device, is_buyer=True)
            buyer_df_with_the_key = self.buyer_dataset[
                list(self.buyer_dataset.columns.difference(join_keys)) + [join_key]]
            buyer_sketch = BuyerSketch(
                buyer_df_with_the_key,
                [join_key],
                join_key_domains,
                buyer_sketch_base,
                target_feature,
                self.device,
                fit_by_residual,
            )
            self.buyer_sketches[join_key] = {
                "id": self.buyer_id,
                "join_key": join_key,
                "join_key_domain": join_key_domains,
                "buyer_sketch": buyer_sketch,
            }
            buyer_sketch.register_this_buyer(fit_by_residual=fit_by_residual)

        # Keep the full-feature buyer frame around for downstream callers — when
        # fitting by residual, ``self.buyer_dataset`` has been reduced to
        # ``[join_keys, residuals]`` and is not useful for feature inspection.
        self.buyer_id_to_df_and_name.append(
            {"name": target_feature,
             "dataframe": self.buyer_dataset_for_residual if fit_by_residual else self.buyer_dataset})
        self.buyer_id += 1
        return self.buyer_id - 1

    def get_buyer_sketch(self, buyer_id):
        return self.buyer_sketches[buyer_id]["buyer_sketch"]

    def get_seller_sketch_by_keys(self, join_key, seller_id):
        return self.seller_sketches[join_key][seller_id]["seller_sketch"]

    def get_seller_sketch_base_by_keys(self, join_key):
        return self.seller_sketches[join_key]["sketch_base"]

    def get_buyer_sketch_by_keys(self, join_key):
        return self.buyer_sketches[join_key]["buyer_sketch"]

    def set_buyer_id(self, buyer_id):
        self.buyer_id = buyer_id

    def reset_buyer_sketches(self):
        self.buyer_sketches = {}

    def reset_buyer_id_to_df_and_name(self):
        self.buyer_id_to_df_and_name = []


class SearchEngine:
    def __init__(self, data_market: DataMarket, fit_by_residual=False):
        self.augplan = []
        self.augplan_acc = []
        self.aug_seller_feature_ind = {}
        self.buyer_target = data_market.buyer_target_feature
        # When fitting by residual, ``buyer_dataset`` has been reduced to the
        # residuals frame; the full feature list lives on the cached residual copy.
        if fit_by_residual and data_market.buyer_dataset_for_residual is not None:
            self.buyer_features = data_market.buyer_dataset_for_residual.columns
        else:
            self.buyer_features = data_market.buyer_dataset.columns
        self.buyer_dataset = None
        self.buyer_sketches = {}
        self.seller_sketches = {}
        self.fit_by_residual = fit_by_residual
        self.data_market = data_market
        self.seller_aggregated = {}
        self.unusable_features = {}

    def search_one_iteration(self):
        best_r_squared = 0
        best_r_squared_ind = -1
        best_batch_id = -1
        best_join_key = None

        self.buyer_sketches = self.data_market.buyer_sketches
        for join_key in self.buyer_sketches.keys():
            buyer_id = self.buyer_sketches[join_key]["id"]
            buyer_sketch = self.buyer_sketches[join_key]["buyer_sketch"]

            buyer_1, buyer_y, buyer_y_y, buyer_x_y = buyer_sketch.get_sketches()
            search_sketch_base = self.data_market.get_seller_sketch_base_by_keys(join_key)

            for batch_id in range(search_sketch_base.get_sketch_loader().get_num_batches()):
                seller_1, seller_x, seller_x_x, _ = search_sketch_base.get_sketch_loader().get_sketches(batch_id)

                if not self.fit_by_residual:
                    d = buyer_y.shape[1]
                    ordered_columns = buyer_sketch.get_base().dfid_feature_mapping[buyer_id]
                    y_ind = ordered_columns.index(buyer_sketch.get_target_feature()["name"])

                    XTX = torch.zeros(seller_x.shape[1], d + 1, d + 1).to(self.data_market.device)
                    XTY = torch.zeros(seller_x.shape[1], d + 1, 1).to(self.data_market.device)
                    c = torch.sum(buyer_1 * seller_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x_x[x_x == 0] = 1
                    y = torch.sum(buyer_y[:, y_ind:y_ind + 1] * seller_1, dim=0)
                    y_y = torch.sum(buyer_y_y[:, y_ind:y_ind + 1] * seller_1, dim=0)
                    TSS = y_y - y * y / c

                    XTX[:, 0, 0] = c
                    XTX[:, 0, 1] = XTX[:, 1, 0] = x
                    XTX[:, 1, 1] = x_x

                    for i in range(d):
                        cur_buyer_y = buyer_y[:, i:i + 1]
                        cur_buyer_y_y = buyer_y_y[:, i:i + 1]
                        cur_x_y = torch.sum(seller_x * cur_buyer_y, dim=0)
                        cur_y_y = torch.sum(cur_buyer_y_y * seller_1, dim=0)
                        cur_y = torch.sum(cur_buyer_y * seller_1, dim=0)
                        cur_y_y[cur_y_y == 0] = 1

                        if i == y_ind:
                            XTY[:, 0, 0] = cur_y
                            XTY[:, 1, 0] = cur_x_y
                        elif i < y_ind:
                            XTX[:, i + 2, i + 2] = cur_y_y
                            XTX[:, 1, i + 2] = XTX[:, i + 2, 1] = cur_x_y
                            XTX[:, 0, i + 2] = XTX[:, i + 2, 0] = cur_y
                        else:
                            XTX[:, i + 1, i + 1] = cur_y_y
                            XTX[:, 1, i + 1] = XTX[:, i + 1, 1] = cur_x_y
                            XTX[:, 0, i + 1] = XTX[:, i + 1, 0] = cur_y

                        for j in range(i + 1, d):
                            x_y_ind = int((2 * d - i - 1) * i / 2 + j - i) - 1
                            x_y_ = torch.sum(buyer_x_y[:, x_y_ind:x_y_ind + 1] * seller_1, dim=0)
                            if i == y_ind:
                                XTY[:, j + 1, 0] = x_y_
                            elif j == y_ind:
                                XTY[:, i + 2, 0] = x_y_
                            elif i > y_ind:
                                XTX[:, i + 1, j + 1] = XTX[:, j + 1, i + 1] = x_y_
                            elif i < y_ind and j > y_ind:
                                XTX[:, i + 2, j + 1] = XTX[:, j + 1, i + 2] = x_y_
                            else:
                                XTX[:, i + 2, j + 2] = XTX[:, j + 2, i + 2] = x_y_

                    inverses = torch.empty_like(XTX)
                    for i in range(len(XTX)):
                        try:
                            inverses[i] = torch.linalg.inv(XTX[i])
                        except RuntimeError:
                            self.unusable_features.setdefault(batch_id, []).append(i)
                            inverses[i] = torch.zeros_like(XTX[i])

                    res = torch.bmm(inverses, XTY).to(self.data_market.device)
                    RSS = y_y
                    for i in range(d + 1):
                        for j in range(d + 1):
                            RSS += res[:, i, 0] * res[:, j, 0] * XTX[:, i, j]
                        RSS -= 2 * res[:, i, 0] * XTY[:, i, 0]
                    r_squared = 1 - RSS / TSS
                else:
                    x_x = torch.sum(seller_x_x * buyer_1, dim=0)
                    x = torch.sum(seller_x * buyer_1, dim=0)
                    c = torch.sum(buyer_1 * seller_1, dim=0)
                    x_y = torch.sum(seller_x * buyer_y, dim=0)
                    y_y = torch.sum(buyer_y_y * seller_1, dim=0)
                    y = torch.sum(buyer_y * seller_1, dim=0)

                    x_mean = x / c
                    y_mean = y / c

                    S_xx = x_x - 2 * x_mean * x + c * x_mean ** 2
                    S_xy = x_y - x_mean * y - x * y_mean + c * x_mean * y_mean

                    slope = S_xy / S_xx
                    intercept = y_mean - slope * x_mean

                    TSS = y_y - 2 * y_mean * y + c * y_mean ** 2
                    RSS = y_y + c * intercept ** 2 + slope ** 2 * x_x - 2 * \
                        (slope * x_y + intercept * y - slope * intercept * x)

                    r_squared = 1 - (RSS / TSS)

                r_squared = torch.where(torch.isnan(r_squared), torch.tensor(float('-inf')), r_squared)
                r_squared = torch.where(r_squared >= 1, torch.tensor(float('-inf')), r_squared)

                if batch_id in self.unusable_features:
                    for singular_ind in self.unusable_features[batch_id]:
                        r_squared[singular_ind] = float('-inf')

                if join_key in self.aug_seller_feature_ind and batch_id in self.aug_seller_feature_ind[join_key]:
                    exclude_indices = self.aug_seller_feature_ind[join_key][batch_id]
                    original_values = r_squared[exclude_indices].clone()
                    r_squared[exclude_indices] = float('-inf')
                    max_r2_index = torch.argmax(r_squared)
                    if r_squared[max_r2_index].item() < -1:
                        r_squared[exclude_indices] = original_values
                        continue
                    r_squared[exclude_indices] = original_values
                else:
                    max_r2_index = torch.argmax(r_squared)

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
            return None, None, None
        return best_join_key, best_r_squared_ind.item(), best_batch_id

    def start(self, iter=2, budget_seconds=None):
        # Anytime budget: stop the greedy search at an iteration boundary once
        # the wall-clock deadline passes (``budget_seconds=None`` disables it).
        # Trajectory emission is intentionally NOT done here: the search plan
        # differs from the augmentation plan that is finally materialized in
        # ``KitanaAugmenter.run`` (some selected features fail to join), so the
        # caller emits the trajectory around the materialization loop where the
        # recorded features match the returned augplan.
        from matryoshka.selection.anytime import BudgetClock
        clock = BudgetClock(budget_seconds).start()
        for i in range(iter):
            join_key, ind, batch_id = self.search_one_iteration()
            if not join_key:
                print("No more good features")
                break

            if join_key not in self.aug_seller_feature_ind:
                self.aug_seller_feature_ind[join_key] = {batch_id: torch.tensor([ind])}
            elif batch_id not in self.aug_seller_feature_ind[join_key]:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.tensor([ind])
            else:
                self.aug_seller_feature_ind[join_key][batch_id] = torch.cat(
                    (self.aug_seller_feature_ind[join_key][batch_id], torch.tensor([ind])))

            seller_id, best_feature = self.data_market.get_seller_sketch_base_by_keys(
                join_key).get_df_by_feature_index(batch_id, ind)
            print(f"The best feature in iter {i} is: {best_feature} with join key {join_key}")

            self.augplan.append((
                seller_id,
                i + 1,
                self.data_market.seller_id_to_df_and_name[seller_id]["name"],
                best_feature,
            ))
            self._update_residual(join_key, seller_id, best_feature)

            # Poll the deadline at the iteration boundary and stop if exceeded.
            if clock.expired:
                break

        buyer_dataset = (self.data_market.buyer_dataset
                        if not self.fit_by_residual
                        else self.data_market.buyer_dataset_for_residual)
        return self.augplan, self.data_market.augplan_acc, buyer_dataset

    def _update_residual(self, join_key, seller_id, best_feature):
        buyer = self.data_market.buyer_id_to_df_and_name[0]["dataframe"]
        if self.fit_by_residual:
            buyer = self.data_market.buyer_dataset_for_residual
        seller_df = self.data_market.get_seller_sketch_by_keys(
            join_key=join_key, seller_id=seller_id).get_df()[[join_key, best_feature]]

        aggregation_functions = {col: 'mean' for col in seller_df.columns if col != join_key}
        seller_df_agg = seller_df.groupby(join_key).agg(aggregation_functions).reset_index()

        # Normalise join keys to defend against dtype drift between buyer and
        # seller (lake CSVs frequently land with mixed int/str/object types for
        # the same logical key). Upstream Kitana-e2e assumes already-clean keys.
        buyer[join_key] = buyer[join_key].apply(process_key)
        seller_df_agg[join_key] = seller_df_agg[join_key].apply(process_key)

        joined_df = pd.merge(buyer, seller_df_agg, how='left', on=join_key,
                             suffixes=('_KitanaSearchLeft', '_KitanaSearchRight'))
        joined_df = joined_df[[col for col in joined_df.columns if '_KitanaSearchRight' not in col]]
        joined_df.columns = [col.replace('_KitanaSearchLeft', '') for col in joined_df.columns]

        for col in seller_df_agg.columns:
            if col != join_key:
                null_count = joined_df[col].isnull().sum()
                if null_count > 0:
                    joined_df[col].fillna(joined_df[col].mean(), inplace=True)

        updated_buyer = joined_df[list(set(
            [join_key] + list(buyer.columns) + [best_feature] + [self.buyer_target]))]

        buy_keys = self.data_market.buyer_join_keys
        join_key_domains = self.data_market.buyer_sketches[join_key]["join_key_domain"]

        self.data_market.set_buyer_id(0)
        self.data_market.reset_buyer_sketches()
        self.data_market.reset_buyer_id_to_df_and_name()
        self.data_market.register_buyer(updated_buyer, buy_keys,
                                        join_key_domains, self.buyer_target,
                                        fit_by_residual=self.fit_by_residual)
