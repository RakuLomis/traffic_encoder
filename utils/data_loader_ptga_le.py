import torch
from torch.utils.data import Dataset
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data 
from models.FieldEmbedding import FieldEmbedding
from utils.dataframe_tools import protocol_tree, add_root_layer
from utils.data_loader import _preprocess_address
from collections import defaultdict
import torch.nn as nn
from typing import Dict, List, Any
from typing import Optional
import gc


class GNNTrafficDataset(Dataset):
    """
    NNtaset?
    _getitem__taFrameyG?Data)?
    """
    def __init__(self, dataframe: pd.DataFrame, config_path: str, vocab_path: str, 
                #  enabled_layers: List[str] | None, 
                enabled_layers: Optional[List[str]], 
                node_feature_dim: int=128, 
                use_flow_features: bool = False, use_ip_address: bool = True, use_mac_address: bool = True, 
                use_port: bool = True, 
                obfuscation_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        print(f"\nInitializing Hierarchical GNNTrafficDataset (Flow Features: {use_flow_features})...")
        self.use_flow_features = use_flow_features
        self.use_ip_address = use_ip_address 
        self.use_mac_address = use_mac_address 
        self.use_port = use_port 
        self.obfuscation_config = obfuscation_config


        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['field_embedding_config']
        with open(vocab_path, 'r') as f:
            self.vocab_maps = yaml.safe_load(f)
        self.labels = torch.tensor(dataframe['label_id'].values, dtype=torch.long)
        # Keep per-packet flow id for flow-level loss/evaluation aggregation.
        if 'stream_id' in dataframe.columns:
            flow_codes, _ = pd.factorize(
                dataframe['stream_id'].fillna('__MISSING_STREAM__').astype(str),
                sort=False
            )
            self.flow_ids = torch.tensor(flow_codes, dtype=torch.long)
        else:
            self.flow_ids = torch.arange(len(dataframe), dtype=torch.long)
        self.TORCH_LONG_MAX = torch.iinfo(torch.long).max
        # self.decimal_fields = {'tcp.stream'}

        # --- ?! ?!!?---
        
        # 1. ?(YAML type ?'categorical')
        self.port_fields = {'tcp.srcport', 'tcp.dstport'}
        
        # 2. ?(in -> ndex)
        port_bin_vocab = {
            0: 0, # Bin 0: Unknown/NaN/Port 0
            1: 1, # Bin 1: Well-Known (1-1023)
            2: 2, # Bin 2: Registered (1024-49151)
            3: 3  # Bin 3: Ephemeral (49152+)
        }
        
        # 3.  self.vocab_maps
        print(" -> Injecting semantic port binning vocabulary...")
        for field in self.port_fields:
            self.vocab_maps[field] = port_bin_vocab
            
        # --- ?! ?!!?---
        # (?YAML ?tcp.stream ?type ?'numerical')
        self.decimal_fields = {'tcp.stream'} 
        
        # --- [!!  !!] ---

        # --- 2. ?(Schema) --- 
        all_available_fields = set(dataframe.columns)
        eth_fields = {f for f in all_available_fields if f.startswith('eth.')}
        ip_fields = {f for f in all_available_fields if f.startswith('ip.')}
        tcp_core_fields = {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f} 
        # ip_fields = {f for f in all_available_fields if f.startswith('ip.') and 'payload' not in f}

        if not use_mac_address: 
            eth_fields_to_ignore = {'eth.src', 'eth.dst'} 
            eth_fields_cleaned = eth_fields - eth_fields_to_ignore
        else: 
            eth_fields_cleaned = eth_fields

        # ?! ?IP ?!!?
        if not use_ip_address: 
            ip_fields_to_ignore = {'ip.src', 'ip.dst'} 
            ip_fields_cleaned = ip_fields - ip_fields_to_ignore
        else: 
            ip_fields_cleaned = ip_fields

        if not use_port: 
            tcp_core_fields_cleaned = tcp_core_fields - self.port_fields 
            print("Port addresses are not included in this run !!!")
        else: 
            tcp_core_fields_cleaned = tcp_core_fields
        
        # ?
        self.expert_definitions = {
            # 'eth': {f for f in all_available_fields if f.startswith('eth.')},
            'eth': eth_fields_cleaned,
            # 'ip': {f for f in all_available_fields if f.startswith('ip.')},
            'ip': ip_fields_cleaned, 
            # 'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f},
            'tcp_core': tcp_core_fields_cleaned, 
            # 'tcp_core': {f for f in all_available_fields if f.startswith('tcp.') and 'options' not in f 
            #                 and 'payload' not in f
            #                 and 'segment_data' not in f},
            'tcp_options': {f for f in all_available_fields if f.startswith('tcp.options.')},
            'tls_record': {f for f in all_available_fields if f.startswith('tls.record.')},
            'tls_handshake': {f for f in all_available_fields if f.startswith('tls.handshake.')},
            'tls_x509': {f for f in all_available_fields if f.startswith('tls.x509')}
            # ... ?
        }
        # self.flow_feature_names = ['flow_avg_len', 'flow_std_len', 'flow_pkt_count']
        self.flow_feature_names = [f for f in dataframe.columns if f.startswith('flow_')]

        print(f"Now we have flow features: {self.flow_feature_names}.")

        # ADD: for Ablation
        # layer -> experts ?tcp/tls ?
        self.layer_to_experts = {
            "eth": ["eth"],
            "ip": ["ip"],
            "tcp": ["tcp_core", "tcp_options"],
            "tls": ["tls_record", "tls_handshake", "tls_x509"],
        }
        
        # enabled_layers ?["tcp", "tls"]one 
        if enabled_layers is not None:
            enabled = set()
            for layer in enabled_layers:
                enabled.update(self.layer_to_experts.get(layer, []))
        
            # ?expert_definitions
            self.expert_definitions = {
                name: fields
                for name, fields in self.expert_definitions.items()
                if name in enabled
            }

        # --- 3. ?---
        print("Pre-calculating graph structures for each expert...")
        self.expert_graphs = {}
        # DataFrame?
        self.all_feature_cols_to_process = set() 
        
        for name, expert_fields in self.expert_definitions.items():
            # chemataFrame?
            real_nodes_for_expert = sorted(list(expert_fields.intersection(all_available_fields)))
            if not real_nodes_for_expert:
                print(f" -> Warning: expert '{name}' has no matched fields in dataframe. Skipping.")
                continue

            # ?
            ptree = protocol_tree(real_nodes_for_expert)
            add_root_layer(ptree)
            
            # ?
            ptree_nodes = set(ptree.keys())
            for children in ptree.values(): 
                ptree_nodes.update(children)
            
            all_nodes_for_expert = sorted(list(ptree_nodes))
            field_to_node_idx = {n: i for i, n in enumerate(all_nodes_for_expert)}
            
            # ?
            self.expert_graphs[name] = {
                'real_nodes': real_nodes_for_expert, # ?
                'all_nodes': all_nodes_for_expert,  # ??
                'field_to_node_idx': field_to_node_idx,
                'edge_index': self._create_edge_index_from_tree(ptree, field_to_node_idx)
            }
            # ?
            self.all_feature_cols_to_process.update(real_nodes_for_expert)

        if self.use_flow_features: 
            print("Flow features enabled. Adding to processing list.")
            self.all_feature_cols_to_process.update(self.flow_feature_names)
        else:
            print("Flow features disabled.")

        # --- 4. ataFrame ---
        print(f"Pre-processing all {len(self.all_feature_cols_to_process)} columns...")
        self.processed_df = self._preprocess_all(dataframe)
        print("Dataset initialization complete.")

        # ?
        self._preload_to_tensors()
        # Cache batched edge_index by (expert_name, batch_size).
        self._batched_edge_index_cache: Dict[str, Dict[int, torch.Tensor]] = {
            expert_name: {} for expert_name in self.expert_graphs
        }
        print("Dataset initialization complete.")

        # # ==================== ?!  !!?====================
        # print("\nPre-caching all items into RAM. This may take several minutes...")
        # # 
        # # init__?
        # #
        # self.data_cache = []
        # for idx in tqdm(range(len(self.labels)), desc="Caching data items"):
        #     # ?"? ?
        #     self.data_cache.append(self._create_data_dict(idx))

        # print(f"Caching complete. {len(self.data_cache)} items loaded into RAM.")
        # # ====================================================================

        
    def _create_edge_index_from_tree(self, ptree: Dict[str, List[str]], field_to_node_idx: Dict[str, int]) -> torch.Tensor:
        """treedx?"""
        edge_list = []
        for parent, children in ptree.items(): 
            if parent in field_to_node_idx:
                parent_idx = field_to_node_idx[parent]
                for child in children:
                    if child in field_to_node_idx:
                        child_idx = field_to_node_idx[child]
                        edge_list.append([child_idx, parent_idx])
                        edge_list.append([parent_idx, child_idx])
        if not edge_list: 
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
        # Vectorized column preprocessing.
        def bin_port_from_hex(x): # ?!?
            if not pd.notna(x): return 0 # Bin 0
            try:
                port_int = int(str(x).split('.')[0], 16) 
                if 1 <= port_int <= 1023: return 1 # Bin 1
                elif 1024 <= port_int <= 49151: return 2 # Bin 2
                elif port_int >= 49152: return 3 # Bin 3
                else: return 0 # Bin 0
            except ValueError: return 0

        def robust_hex_to_int(x):
            if not pd.notna(x): return 0
            try: return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
            except ValueError: return 0

        def robust_timestamp_to_tsval(x):
            if not pd.notna(x): return 0
            try:
                s = str(x).lower().replace('0x', '')
                if len(s) != 20 or not s.startswith('080a'): return 0
                tsval_hex = s[4:12]
                return int(tsval_hex, 16)
            except (ValueError, TypeError): return 0

        processed_data_dict = {}
        
        # ?
        cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
        for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
            # 1. ?(Flow Stats)
            if field_name in self.flow_feature_names:
                processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
                continue
                
            # 2.  (GNN Node Features)
            if field_name not in self.config:
                continue # ?

            config = self.config[field_name]
            col_data = df[field_name]
            field_type = config['type']

            if field_type in ['address_ipv4', 'address_mac']:
                processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
            elif field_type == 'categorical':
                # ?! ?!!?
                if field_name in self.port_fields and self.use_port:
                    # a. Hex Port -> Bin ID (0, 1, 2, 3)
                    binned_data = col_data.apply(bin_port_from_hex)
                    # b. Map Bin ID -> Embedding Index (?{0:0, 1:1, ...})
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = 0 # (0 ?0)
                    processed_data_dict[field_name] = binned_data.map(vocab_map).fillna(oov_index).astype(np.int32)
                
                # ?!?? ?
                elif field_name in self.vocab_maps:
                    vocab_map = self.vocab_maps[field_name]
                    oov_index = vocab_map.get('__OOV__', 0)
                    processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
                else:
                    pass # (D?serialNumber)

            elif field_type == 'numerical':
                if field_name in self.decimal_fields: # (?tcp.stream)
                    processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
                elif field_name == 'tcp.options.timestamp': # (?080a...)
                    # processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).astype(np.int64)
                    # ?!  !!?
                    # 1. ?
                    # 2. ?np.log1p (log(x+1)) ?
                    # 3. ?float32
                    # processed_data_dict[field_name] = col_data.apply(robust_timestamp_to_tsval).apply(np.log1p).astype(np.float32)
                    values = col_data.apply(robust_timestamp_to_tsval).astype(np.float32)

                    # =============================
                    # IAT / Timestamp Noise
                    # =============================
                    if (
                        self.obfuscation_config
                        and self.obfuscation_config.get("iat_noise") is not None
                    ):
                        sigma = self.obfuscation_config["iat_noise"]
                        values = values + np.random.normal(
                            0,
                            sigma,
                            size=len(values)
                        )
                
                    values = np.log1p(values)
                    processed_data_dict[field_name] = values.astype(np.float32)

                else: # (? ip.len, tcp.len, tsval, tsecr ...)
                    # processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
                    values = col_data.apply(robust_hex_to_int).astype(np.float32)

                    # =============================
                    # Packet Length Obfuscation
                    # =============================
                    if (
                        self.obfuscation_config
                        and field_name == "ip.len"
                        and self.obfuscation_config.get("len_noise") is not None
                    ):
                        noise_level = self.obfuscation_config["len_noise"]
                        noise = np.random.uniform(
                            -noise_level,
                            noise_level,
                            size=len(values)
                        )
                        values = values * (1 + noise)

                    processed_data_dict[field_name] = values.astype(np.float32)

            else:
                 print(f"{field_name}: {field_type}")
            
            
        return pd.DataFrame(processed_data_dict)

    # def _preprocess_all(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     ?
    #     _init__taFrame?
    #     """
    #     processed_data_dict = {}
        
    #     # ?
    #     cols_to_process = sorted(list(self.all_feature_cols_to_process.intersection(df.columns)))
        
    #     for field_name in tqdm(cols_to_process, desc="Pre-processing columns"):
    #         # 1. ?(Flow Stats)
    #         if field_name in self.flow_feature_names:
    #             processed_data_dict[field_name] = pd.to_numeric(df[field_name], errors='coerce').fillna(0.0).astype(np.float32)
    #             continue
                
    #         # 2.  (GNN Node Features)
    #         if field_name not in self.config:
    #             continue # ?
            
    #         config = self.config[field_name]
    #         col_data = df[field_name]

    #         # if/elif?()
    #         if config['type'] in ['address_ipv4', 'address_mac']:
    #             field_type = config['type']
    #             processed_data_dict[field_name] = col_data.apply(lambda x: _preprocess_address(x, field_type))
            
    #         elif field_name in self.vocab_maps:
    #             vocab_map = self.vocab_maps[field_name]
    #             oov_index = vocab_map.get('__OOV__', 0) # OV=0
    #             # processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(int)
    #             processed_data_dict[field_name] = col_data.fillna('__OOV__').astype(str).str.lower().str.replace('0x','').map(vocab_map).fillna(oov_index).astype(np.int32)
            
    #         # elif field_name == 'tcp.stream': #  decimal_fields
    #         #     processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
    #         # ?self.decimal_fields ?
    #         elif field_name in self.decimal_fields: 
    #             # processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(int)
    #             processed_data_dict[field_name] = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(np.int32)
            
    #         elif config['type'] in ['categorical', 'numerical']:
    #             def robust_hex_to_int(x):
    #                 if not pd.notna(x): return 0
    #                 try:
    #                     return min(int(str(x).split('.')[0], 16), self.TORCH_LONG_MAX)
    #                 except ValueError:
    #                     return 0
    #             # processed_data_dict[field_name] = col_data.apply(robust_hex_to_int)
    #             processed_data_dict[field_name] = col_data.apply(robust_hex_to_int).astype(np.int32)
            
    #     return pd.DataFrame(processed_data_dict)

    def __len__(self):
        return len(self.labels)

    def _get_batched_edge_index(
        self,
        base_edge_index: torch.Tensor,
        num_nodes_per_graph: int,
        batch_size: int,
    ) -> torch.Tensor:
        if base_edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)

        offsets = (
            torch.arange(batch_size, dtype=base_edge_index.dtype)
            .view(batch_size, 1, 1)
            * num_nodes_per_graph
        )
        repeated = base_edge_index.unsqueeze(0) + offsets
        return repeated.permute(1, 0, 2).reshape(2, -1).contiguous()

    def collate_from_index(self, batch_indices: List[int]) -> Dict[str, Any]:
        """
        Build one batched `dict[str, Data]` from index list.
        """
        if len(batch_indices) == 0:
            return {}

        idx = torch.as_tensor(batch_indices, dtype=torch.long)
        batch_size = idx.numel()
        labels = self.labels.index_select(0, idx)
        flow_ids = self.flow_ids.index_select(0, idx)
        batched: Dict[str, Any] = {}

        for expert_name, graph_info in self.expert_graphs.items():
            expert_tensors = self.tensor_cache[expert_name]
            feature_dict: Dict[str, torch.Tensor] = {}

            for field_name in graph_info['all_nodes']:
                feature_dict[field_name] = expert_tensors[field_name].index_select(0, idx)

            num_nodes_per_graph = len(graph_info['all_nodes'])
            edge_cache = self._batched_edge_index_cache[expert_name]
            batched_edge_index = edge_cache.get(batch_size)
            if batched_edge_index is None:
                batched_edge_index = self._get_batched_edge_index(
                    graph_info['edge_index'],
                    num_nodes_per_graph=num_nodes_per_graph,
                    batch_size=batch_size,
                )
                edge_cache[batch_size] = batched_edge_index

            batch_vec = torch.arange(batch_size, dtype=torch.long).repeat_interleave(
                num_nodes_per_graph
            )

            batched[expert_name] = Data(
                edge_index=batched_edge_index,
                batch=batch_vec,
                y=labels,
                num_nodes=batch_size * num_nodes_per_graph,
                **feature_dict,
            )

        if self.use_flow_features:
            flow_cols = [
                self.flow_tensor_cache[f].index_select(0, idx)
                for f in self.flow_feature_names
            ]
            batched['flow_stats'] = torch.stack(flow_cols, dim=1).contiguous()
        batched['flow_ids'] = flow_ids

        return batched


    def _preload_to_tensors(self):
        """
        ?Pandas DataFrame ?PyTorch Tensors?
        ?__getitem__ ?CPU ?
        """
        print("Converting dataframe columns to PyTorch tensors (Vectorization)...")
        self.tensor_cache = {} # ? {expert_name: {field_name: Tensor}}
        num_samples = len(self.processed_df)
        
        for expert_name, graph_info in self.expert_graphs.items():
            self.tensor_cache[expert_name] = {}
            all_nodes_list = graph_info['all_nodes']
            real_nodes_set = set(graph_info['real_nodes'])
            
            for field_name in all_nodes_list:
                # 1.  dtype ?shape
                config = self.config.get(field_name, {})
                config_type = config.get('type', 'categorical')
                
                # --- ?A: ?---
                if field_name in real_nodes_set and field_name in self.processed_df.columns:
                    col_data = self.processed_df[field_name]
                    
                    if config_type in ['address_ipv4', 'address_mac']:
                        # ?list?
                        # ?Series of Lists ?Tensor [N, 4] or [N, 6]
                        # ?
                        try:
                            # ?col_data ?list ?
                            data_tensor = torch.tensor(col_data.tolist(), dtype=torch.long)
                        except:
                            # ?
                            dim = 4 if config_type == 'address_ipv4' else 6
                            data_tensor = torch.zeros((num_samples, dim), dtype=torch.long)

                    elif config_type == 'numerical':
                        # ?-> float32 [N]
                        values = col_data.fillna(0).values.astype(np.float32)
                        data_tensor = torch.from_numpy(values)
                        
                    else:
                        # ?-> long [N]
                        values = col_data.fillna(0).values.astype(np.int64)
                        data_tensor = torch.from_numpy(values)
                
                # --- ?B: ?(? Tensor) ---
                else:
                    if config_type == 'address_ipv4':
                        data_tensor = torch.zeros((num_samples, 4), dtype=torch.long)
                    elif config_type == 'address_mac':
                        data_tensor = torch.zeros((num_samples, 6), dtype=torch.long)
                    elif config_type == 'numerical':
                        data_tensor = torch.zeros((num_samples,), dtype=torch.float32)
                    else:
                        data_tensor = torch.zeros((num_samples,), dtype=torch.long)

                # 
                self.tensor_cache[expert_name][field_name] = data_tensor
        
        # ?Flow Features (?
        if self.use_flow_features:
            self.flow_tensor_cache = {}
            for f in self.flow_feature_names:
                col_data = self.processed_df[f]
                values = col_data.fillna(0).values.astype(np.float32)
                self.flow_tensor_cache[f] = torch.from_numpy(values)
                
        print("Vectorization complete. DataFrame memory can be released.")
        # ?DataFrame ?(?Tensor ?
        del self.processed_df
    def __getitem__(self, idx) -> int:
        """
        Return index only. Batch assembly is handled by `collate_from_index`.
        """
        return int(idx)
    
    # def __getitem__(self, idx) -> Dict[str, Any]:
    #     """
    #     ?
    #     ?
    #     """
    #     return self.data_cache[idx]


