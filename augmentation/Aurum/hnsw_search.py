import numpy as np
import random
import pickle
import time
import hnswlib
import os

from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm


class HNSWSearcher(object):
    def __init__(self, table_path, index_path, scale, search_mode='join', random_seed=42):
        """
        Initialize HNSW searcher.
        
        Args:
            table_path: Path to pickled table embeddings
            index_path: Path to HNSW index (will load if exists, otherwise build and save)
            scale: Percentage of tables to use (for scalability experiments)
            search_mode: 'join' or 'union' - affects scoring strategy
            random_seed: Random seed for deterministic table sampling
        """
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)
        # For scalability experiments: load a percentage of tables
        # Set all random seeds for deterministic behavior
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()
        self.vec_dim = len(self.tables[1][1][0])
        self.search_mode = search_mode     

        index_start_time = time.time()
        self.index = hnswlib.Index(space='cosine', dim=self.vec_dim)
        self.index.set_num_threads(1)
        self.all_columns, self.col_table_ids = self._preprocess_table_hnsw()
        
        if not os.path.exists(index_path):
            # Build index from scratch
            print(f"Index file not found at {index_path}, building from scratch...")
            self.index.init_index(max_elements=len(self.all_columns), ef_construction=100, M=32, random_seed=random_seed)
            self.index.set_ef(10)
            self.index.add_items(self.all_columns)
            
            # Save the index
            index_dir = os.path.dirname(index_path)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir)
            self.index.save_index(index_path)
            print(f"Index saved to {index_path}")
            print("--- Indexing Time: %s seconds ---" % (time.time() - index_start_time))
        else:
            # Load pre-built index
            print(f"Loading pre-built index from {index_path}...")
            self.index.load_index(index_path, max_elements=len(self.all_columns))
            self.index.set_ef(10)
            print("--- Index Loading Time: %s seconds ---" % (time.time() - index_start_time))

    
    def topk(self, enc, query, K, N=5, threshold=0.6, max_join_cols=3):
        """
        Find top-K candidate tables.
        
        Args:
            enc: Encoder type
            query: Query table (name, embeddings)
            K: Number of results to return
            N: Number of neighbors to retrieve per column
            threshold: Similarity threshold for column matching
            max_join_cols: For join mode, maximum number of columns to consider as join keys
        
        Returns:
            scores: List of (score, column_pairs, table_name) tuples
            scoreLength: Total number of candidates
        """
        # Note: N is the number of columns retrieved from the index
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        candidates = self._find_candidates(query_cols, N)
        if enc == 'sato':
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in candidates:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else: # encoder is sherlock
            scores = []
            num_query_cols = len(query[1])
            # Make max_join_cols relative to query width: at most max_join_cols OR 50% of query columns
            effective_max_join = max(max_join_cols, int(num_query_cols * 0.5))
            for table in candidates:
                total_score, column_pairs = self._verify(query[1], table[1], threshold)
                
                if self.search_mode == 'join':
                    # For joins: prefer tables with few high-quality matches
                    num_matches = len(column_pairs)
                    if num_matches == 0 or num_matches > effective_max_join:
                        # Skip tables with no matches or too many matches (likely union candidates)
                        continue
                    # Score based on average quality of matches (not total)
                    avg_score = total_score / num_matches if num_matches > 0 else 0
                    # Boost score for tables with 1-2 matches (typical join scenario)
                    if num_matches <= 2:
                        avg_score *= 1.5
                    scores.append((avg_score, column_pairs, table[0]))
                else:  # union mode
                    # For unions: prefer tables with many matches (original behavior)
                    scores.append((total_score, column_pairs, table[0]))
                    
        # Sort by score (descending), then by table name (ascending) for deterministic ordering
        scores.sort(key=lambda x: (-x[0], x[2]))
        scoreLength = len(scores)
        return scores[:K], scoreLength
    
    def _preprocess_table_hnsw(self):
        all_columns = []
        col_table_ids = []
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        return all_columns, col_table_ids
    
    def _find_candidates(self,query_cols, N):
        table_subs = set()
        labels, _ = self.index.knn_query(query_cols, k=N)
        for result in labels:
            # result: list of subscriptions of column vector
            for idx in result:
                table_subs.add(self.col_table_ids[idx])
        candidates = []
        for tid in sorted(table_subs):  # Sort for deterministic ordering
            candidates.append(self.tables[tid])
        return candidates
    
    def _cosine_sim(self, vec1, vec2):
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

    def _verify(self, table1, table2, threshold):
            score = 0.0
            nrow = len(table1)
            ncol = len(table2)
            union_column = []
            graph = np.zeros(shape=(nrow,ncol),dtype=float)
            for i in range(nrow):
                for j in range(ncol):
                    sim = self._cosine_sim(table1[i],table2[j])
                    if sim > threshold:
                        graph[i,j] = sim
                        union_column.append((i, j, sim))

            max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
            m = Munkres()
            indexes = m.compute(max_graph)
            for row,col in indexes:
                score += graph[row,col]
            return score, union_column