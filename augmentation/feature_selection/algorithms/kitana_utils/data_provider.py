import pandas as pd
import numpy as np
import math



class PrepareData():
    def __init__(self, data_path: str, join_keys: list, from_disk: bool, df: pd.DataFrame = None, sep: str = ",", schema_mapping: dict = {}):
        if from_disk:
            self.data = pd.read_csv(data_path, sep=sep)
        else:
            self.data = df
        if schema_mapping != {}:
            self.data.rename(columns=schema_mapping, inplace=True)
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        self.join_keys = []
        self.join_keys_in_string = []
        self.has_key = self.check_join_keys(join_keys=join_keys)

        self.construct_join_keys()
    
    def get_data(self):
        return self.data
    
    def set_data(self, data: pd.DataFrame):
        self.data = data
    
    def see_data(self):
        print(self.data.head())

    def get_features(self):
        return self.data.columns
    
    def see_features(self):
        print(self.data.columns)
    
    def get_join_keys(self):
        return self.join_keys_in_string
    
    def cut_data_by_features(self, features: list):
        return self.data[features]
    
    def cut_data_by_join_keys(self, join_keys: list):
        return self.data[join_keys]
    
    def data_cleaning(self):
        
        def get_num_cols(df, join_keys):
            def is_numeric(val):
                # Check for NaN (pandas NaN or math NaN)
                if pd.isna(val) or (isinstance(val, float) and math.isnan(val)):
                    return False
                # Check for numeric types including numpy types
                return isinstance(val, (int, float, complex, np.integer, np.floating)) and not isinstance(val, bool)

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            num_cols = [col for col in df.columns if is_numeric(df[col].iloc[0])]
            display_cols = [col for col in df.columns]
            for col in df.columns:
                nan_fraction = df[col].apply(lambda x: x == '' or pd.isna(x)).mean()
                # print(nan_fraction, col)
                if nan_fraction > 0.4:
                    display_cols.remove(col)

            # Check if the first row has any NaN values
            df.fillna(0, inplace=True)

            for col in num_cols[:]:  # Iterate over a copy of num_cols
                has_string = df[col].apply(lambda x: isinstance(x, str)).any()

                if has_string:
                    # Calculate the fraction of non-numeric (including NaN) entries
                    non_numeric_fraction = df[col].apply(lambda x: not is_numeric(x)).mean()

                    if non_numeric_fraction > 0.5:
                        # Remove the column from num_cols if more than half entries are non-numeric
                        num_cols.remove(col)
                    else:
                        # Replace non-numeric entries with NaN and then fill them with the mean
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col].fillna(df[col].mean(), inplace=True)
            if join_keys.difference(set(list(df.columns))): 
                return [], None
            else:
                for ele in join_keys.difference(set(display_cols)):
                    display_cols = [ele] + display_cols
            return num_cols
        num_cols = get_num_cols(self.data, set(self.join_keys_in_string))


        return num_cols

    def check_join_keys(self, join_keys: list):
        """
        join_keys is expected to be a list of list of strings (e.g. [["country", "year"], ["country"], ["year"]])
        for each list, we need to check if it is a subset of the column names.
        If not, we do not add the join_keys
        If yes, we add the join_keys
        """
        for join_key in join_keys:
            if set(join_key).issubset(set(self.data.columns)):
                self.join_keys.append(join_key)

        if self.join_keys == []:
            return False
        return True
    
    def construct_join_keys(self):
        """
        For all combination of join keys, we would make a new column that contains both of the values
        Like if we have join_keys = [["country", "year"], ["country"], ["year"]], then we would eventually have
        ["country_year", "country", "year"] as the join keys. All the other columns would be the same.
        """
        for join_key in self.join_keys:
            if len(join_key) > 1:
                # If we have value that is not string, we should convert it to string

                self.data["_".join(join_key)] = self.data[join_key].astype(str).apply(lambda x: "_".join(x), axis=1)
                self.join_keys_in_string.append("_".join(join_key))
            else:
                self.join_keys_in_string.append(join_key[0])
        return self.data


class PrepareBuyer(PrepareData):
    def __init__(self, data_path: str, join_keys: list, target_feature: str, features: list = [], one_target_feature: bool = True, from_disk: bool = True, buyer_df: pd.DataFrame = None, need_to_clean_data: bool = True, sep: str = ","):
        super().__init__(data_path, join_keys, from_disk=from_disk, df = buyer_df, sep=sep)
        self.from_disk = from_disk
        self.target_feature = target_feature

        if need_to_clean_data:
            num_cols = self.data_cleaning()
        else:
            num_cols = features

        self.data = self.data[list(set(self.join_keys_in_string).union(set(num_cols)).union(set([self.target_feature])))]
        self.one_target_feature = one_target_feature
        if self.one_target_feature:
            self.data = self.data[list(set(self.join_keys_in_string).union(set([target_feature])))]
        self.buyer_key_domain = {}
        self.calculate_buyer_key_domain()
    
    def calculate_buyer_key_domain(self):
        for join_key in self.join_keys_in_string:
            self.buyer_key_domain[join_key] = set(self.data[join_key])
        return self.buyer_key_domain


        

class PrepareSeller(PrepareData):
    def __init__(self, data_path: str, join_keys: list, features: list = [], from_disk: bool = True, seller_df: pd.DataFrame = None, need_to_clean_data: bool = True, sep: str = ",", schema_mapping: dict = {}):
        super().__init__(data_path, join_keys, from_disk=from_disk, df = seller_df, sep=sep, schema_mapping=schema_mapping)
        self.key_domain_recorded = False
        self.join_key_domains = {} # {join_key: {set of values}}
        if need_to_clean_data:
            num_cols = self.data_cleaning()
        else:
            num_cols = features
        self.data = self.data[list(set(self.join_keys_in_string).union(set(num_cols)))]
    
    def get_record_status(self):
        return self.key_domain_recorded
    
    def set_join_key_domains(self, join_key: str, domain: set):
        self.join_key_domains[join_key] = domain.union(self.data[join_key])

    def get_join_key_domains(self, join_key: str):
        return self.join_key_domains[join_key]

    def record_join_key_domains(self):
        for join_key in self.join_keys_in_string:
            self.set_join_key_domains(join_key, set(self.data[join_key]))
        self.key_domain_recorded = True
        return self.join_key_domains

class PrepareSellers():
    def __init__(self):
        self.sellers = {}

        self.join_keys = []
        self.join_key_domains = {} # {join_key: {set of values}}


    def get_sellers(self):
        return self.sellers
    
    def see_sellers(self):
        print("Sellers:")
        for seller_name in self.sellers:
            print(f"seller: {seller_name}")
            self.sellers[seller_name].see_data()
    
    def add_sellers(self, seller_name: str, seller: PrepareSeller, buyer: PrepareBuyer):
        if seller.has_key:
            # for join_key_in_string in seller.get_join_keys():
            #     seller_keys_set = set(seller.data[join_key_in_string])
            #     buyer_keys_set = set(buyer.buyer_key_domain[join_key_in_string])
            #     print(f"Seller keys set: {list(seller_keys_set)[:5]}")
            #     print(f"Buyer keys set: {list(buyer_keys_set)[:5]}")

            #     intersection = seller_keys_set.intersection(buyer_keys_set)
            #     if not intersection:
            #         # We should delete this key in string from the join keys by finding the index and deleting it
            #         index = seller.join_keys_in_string.index(join_key_in_string)
            #         seller.join_keys_in_string.pop(index)

            #         # We should also delete the join key column from the data
            #         seller.data.drop(join_key_in_string, axis=1, inplace=True)
            # print(f"After checking intersection, seller join keys are: {seller.join_keys_in_string}")
            # if seller.join_keys_in_string == []:
            #     seller.has_key = False
            # else:
                self.sellers[seller_name] = seller
                self.join_keys = list(set(self.join_keys).union(set(seller.get_join_keys())))
                self.update_domain(seller)
                return True
        print(f"seller: {seller_name} does not have the corresponding join keys")
        return False

    def add_seller_by_path(self, data_path: str, join_keys: list, buyer: PrepareBuyer, features: list, need_to_clean_data: bool = True, sep: str = ",", schema_mapping: dict = {}):
        seller_name = data_path.split("/")[-1].split(".")[0]
        self.add_sellers(seller_name, PrepareSeller(data_path, join_keys, features = features, need_to_clean_data=need_to_clean_data, sep=sep, schema_mapping=schema_mapping), buyer)

    def get_domain(self, join_key: str):
        return self.join_key_domains[join_key]
    
    def update_domain(self, seller: PrepareSeller):
        seller_join_key_domains = seller.record_join_key_domains()
        for join_key in seller_join_key_domains:
            if join_key in self.join_key_domains:
                self.join_key_domains[join_key] = self.join_key_domains[join_key].union(seller_join_key_domains[join_key])
            else:
                self.join_key_domains[join_key] = seller_join_key_domains[join_key]

        # update the seller join key domains
        for seller_name in self.sellers:
            seller = self.sellers[seller_name]
            seller_join_keys = seller.get_join_keys()
            # We update the join key domains for each seller based on the join keys of the sellers
            for join_key in seller_join_keys:
                seller.set_join_key_domains(join_key, self.get_domain(join_key))
    


    def get_seller_names(self):
        return list(self.sellers.keys())

class PrepareBuyerSellers():
    def __init__(self, need_to_clean_data: bool = True, buyer_sep: str = ",", seller_sep: str = ","):
        self.buyer = None
        self.buyer_added = False
        self.need_to_clean_data = need_to_clean_data
        self.sellers = PrepareSellers()
        self.buyer_sep = buyer_sep
        self.seller_sep = seller_sep

    def get_buyer(self):
        return self.buyer
    
    def get_sellers(self):
        return self.sellers
    
    def add_buyer(self, buyer: PrepareBuyer):
        self.buyer_added = True
        self.buyer = buyer
    
    def add_seller(self, seller_name: str, seller: PrepareSeller):
        if not self.buyer_added:
            raise Exception("Buyer has not been added yet")
        self.sellers.add_sellers(seller_name=seller_name, seller=seller, buyer= self.buyer)
    
    def add_buyer_by_path(self, data_path: str, join_keys: list, buyer_features: list, target_feature: str, sep: str = ","):
        self.add_buyer(PrepareBuyer(data_path, join_keys, target_feature, features = buyer_features, one_target_feature=False, need_to_clean_data=self.need_to_clean_data, sep=sep))

    def add_seller_by_path(self, data_path: str, join_keys: list, seller_features: list, schema_mapping: dict = {}):
        self.sellers.add_seller_by_path(data_path, join_keys, self.buyer, features = seller_features, need_to_clean_data=self.need_to_clean_data, sep=self.seller_sep, schema_mapping=schema_mapping)

    def get_domain(self):
        return self.sellers.join_key_domains    

    def get_domain_by_join_key(self, join_key: str):
        return self.sellers.get_domain(join_key)

    def get_join_keys(self):
        return self.sellers.join_keys

    def get_join_key_domains(self):
        return self.sellers.join_key_domains

    def get_seller_join_key_domains(self):
        return self.sellers.get_sellers()

    def get_buyer_join_keys(self):
        return self.buyer.get_join_keys()
    
    def see_buyer_data(self):
        self.buyer.see_data()

    def see_seller_data(self):
        self.sellers.see_sellers()

    def get_buyer_data(self):
        return self.buyer.get_data()

    def get_seller_data(self):
        return self.sellers.get_sellers()

    def get_buyer_features(self):
        return self.buyer.get_features()

    def get_seller_features(self):
        return self.sellers.get_sellers()

    def buyer_cut_data_by_features(self, features: list):
        return self.buyer.cut_data_by_features(features)

    def buyer_cut_data_by_join_keys(self, join_keys: list):
        return self.buyer.cut_data_by_join_keys(join_keys)
    

if __name__ == "__main__":
    # now we test if we can successfully prepare the data
    prepare_data = PrepareBuyerSellers()
    join_keys = ["country", "year"]
    prepare_data.add_buyer_by_path(BUYER_DATA_PATH, join_keys, "suicides_no")
    for seller_path in SELLER_DATA_PATHS:
        print("Now processing seller data: ", seller_path)
        prepare_data.add_seller_by_path(seller_path, join_keys)
    print(prepare_data.get_buyer_data().head())
    prepare_data.see_seller_data()
