import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
from typing import Dict, List, Union
import re

class TrafficFeatureEmbedder:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {
            'ip': OneHotEncoder(sparse_output=False),
            'port': OneHotEncoder(sparse_output=False),
            'proto': OneHotEncoder(sparse_output=False),
            'tls_content_type': OneHotEncoder(sparse_output=False),
            'tls_version': OneHotEncoder(sparse_output=False),
            'tls_handshake_type': OneHotEncoder(sparse_output=False),
            'tls_cipher_suite': OneHotEncoder(sparse_output=False),
            'tls_comp_method': OneHotEncoder(sparse_output=False),
            'tls_extension_type': OneHotEncoder(sparse_output=False),
            'tls_sig_hash_alg': OneHotEncoder(sparse_output=False)
        }
        
    def _normalize_numerical(self, df: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """标准化数值型特征"""
        return self.scaler.fit_transform(df[columns])
    
    def _encode_categorical(self, df: pd.DataFrame, columns: List[str], encoder_name: str) -> np.ndarray:
        """对类别型特征进行one-hot编码"""
        return self.encoders[encoder_name].fit_transform(df[columns].values.reshape(-1, 1))
    
    def _encode_ip(self, ip_series: pd.Series) -> np.ndarray:
        """将IP地址转换为数值向量"""
        ip_parts = ip_series.str.split('.')
        ip_matrix = np.array([list(map(int, parts)) for parts in ip_parts])
        return ip_matrix / 255.0
    
    def _encode_mac(self, mac_series: pd.Series) -> np.ndarray:
        """将MAC地址转换为数值向量"""
        mac_clean = mac_series.str.replace(':', '')
        mac_matrix = np.array([list(map(lambda x: int(x, 16), mac_clean))])
        return mac_matrix / 255.0
    
    def _encode_hex_string(self, hex_series: pd.Series) -> np.ndarray:
        """将十六进制字符串转换为数值向量"""
        hex_values = hex_series.apply(lambda x: int(x, 16) if pd.notna(x) else 0)
        return hex_values.values.reshape(-1, 1) / 65535.0  # 归一化到[0,1]区间
    
    def _encode_timestamp(self, timestamp_series: pd.Series) -> np.ndarray:
        """将时间戳转换为数值向量"""
        # 假设时间戳格式为"YYMMDDHHMMSSZ"
        timestamps = timestamp_series.apply(lambda x: pd.to_datetime(x, format='%y%m%d%H%M%SZ') if pd.notna(x) else pd.NaT)
        unix_timestamps = timestamps.astype(np.int64) // 10**9
        return unix_timestamps.values.reshape(-1, 1)
    
    def _encode_certificate(self, cert_data: pd.DataFrame) -> np.ndarray:
        """处理证书相关字段"""
        # 处理证书版本
        version = cert_data['tls.x509af.version'].values.reshape(-1, 1)
        
        # 处理序列号
        serial = cert_data['tls.x509af.serialNumber'].apply(
            lambda x: int(x, 16) if pd.notna(x) else 0
        ).values.reshape(-1, 1)
        
        # 处理有效期
        not_before = self._encode_timestamp(cert_data['tls.x509af.notBefore'])
        not_after = self._encode_timestamp(cert_data['tls.x509af.notAfter'])
        
        # 处理密钥用法
        key_usage = cert_data[[
            'tls.x509ce.KeyUsage.digitalSignature',
            'tls.x509ce.KeyUsage.contentCommitment',
            'tls.x509ce.KeyUsage.keyEncipherment',
            'tls.x509ce.KeyUsage.dataEncipherment',
            'tls.x509ce.KeyUsage.keyAgreement',
            'tls.x509ce.KeyUsage.keyCertSign',
            'tls.x509ce.KeyUsage.cRLSign',
            'tls.x509ce.KeyUsage.encipherOnly',
            'tls.x509ce.KeyUsage.decipherOnly'
        ]].fillna(0).values
        
        return np.concatenate([version, serial, not_before, not_after, key_usage], axis=1)
    
    def embed_features(self, df_dict: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """整合所有特征embedding"""
        features_list = []
        
        # 1. 基础网络特征
        base_df = df_dict['0.csv']
        numerical_cols = ['ip.ttl', 'ip.len', 'tcp.window_size_value', 
                         'tcp.len', 'tcp.hdr_len']
        features_list.append(self._normalize_numerical(base_df, numerical_cols))
        
        # 2. IP和MAC地址
        features_list.append(self._encode_ip(base_df['ip.src']))
        features_list.append(self._encode_ip(base_df['ip.dst']))
        features_list.append(self._encode_mac(base_df['eth.src']))
        features_list.append(self._encode_mac(base_df['eth.dst']))
        
        # 3. 端口和协议
        features_list.append(self._encode_categorical(
            base_df, ['tcp.srcport', 'tcp.dstport'], 'port'))
        features_list.append(self._encode_categorical(
            base_df, ['ip.proto'], 'proto'))
        
        # 4. TCP选项
        tcp_options_df = df_dict['2.csv']
        features_list.append(self._encode_hex_string(tcp_options_df['tcp.options']))
        
        # 5. TLS记录层
        tls_record_df = df_dict['5.csv']
        features_list.append(self._encode_categorical(
            tls_record_df, ['tls.record.content_type'], 'tls_content_type'))
        
        tls_version_df = df_dict['6.csv']
        features_list.append(self._encode_categorical(
            tls_version_df, ['tls.record.version'], 'tls_version'))
        features_list.append(self._encode_hex_string(tls_version_df['tls.record.length']))
        
        # 6. TLS握手层
        tls_handshake_df = df_dict['7.csv']
        features_list.append(self._encode_categorical(
            tls_handshake_df, ['tls.handshake.type'], 'tls_handshake_type'))
        features_list.append(self._encode_hex_string(tls_handshake_df['tls.handshake.length']))
        
        # 7. TLS扩展
        tls_ext_df = df_dict['8.csv']
        features_list.append(self._encode_categorical(
            tls_ext_df, ['tls.handshake.extension.type'], 'tls_extension_type'))
        
        # 8. 证书信息
        if '16.csv' in df_dict:
            cert_features = self._encode_certificate(df_dict['16.csv'])
            features_list.append(cert_features)
        
        # 合并所有特征
        all_features = np.concatenate(features_list, axis=1)
        return torch.FloatTensor(all_features)

class TrafficEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], output_dim: int = 64):
        super(TrafficEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

def load_discrete_data(base_path: str) -> Dict[str, pd.DataFrame]:
    """加载所有离散的CSV文件"""
    df_dict = {}
    for i in range(21):  # 0-20.csv
        file_path = f"{base_path}/discrete/{i}.csv"
        try:
            df_dict[f"{i}.csv"] = pd.read_csv(file_path)
        except FileNotFoundError:
            continue
    return df_dict

# 使用示例
if __name__ == "__main__":
    # 加载数据
    base_path = "Data/Test/merge_tls_test_01"
    df_dict = load_discrete_data(base_path)
    
    # 初始化embedder
    embedder = TrafficFeatureEmbedder()
    
    # 获取特征tensor
    features = embedder.embed_features(df_dict)
    
    # 创建编码器模型
    input_dim = features.shape[1]
    encoder = TrafficEncoder(input_dim)
    
    # 获取编码后的特征
    encoded_features = encoder(features)
    print(f"Original feature shape: {features.shape}")
    print(f"Encoded feature shape: {encoded_features.shape}") 