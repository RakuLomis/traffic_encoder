import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
from typing import Dict, List, Union
import re

class TrafficFeatureEmbedder: 
    """
    This class considers features of the whole original dataset before truncating into blocks. 
    So it takes the NaN values into account. 
    """
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
        """
        Normalize numerical features. 
        """
        # Ensure all columns are numerical, handle hexadecimal strings and floating point numbers
        numeric_data = df[columns].apply(lambda x: x.apply( # x means the series of a column 
            lambda val: int(str(val), 16) if pd.notna(val) and isinstance(val, str) and val.isalnum() else float(val) if pd.notna(val) else 0
        ))
        # Fill missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        return self.scaler.fit_transform(numeric_data)
    
    def _encode_categorical(self, df: pd.DataFrame, columns: List[str], encoder_name: str) -> np.ndarray:
        """
        Encode categorical features by one-hot encoding. 
        """
        def convert_value(val):
            if pd.isna(val):
                return '0'
            if isinstance(val, str) and val.isalnum():
                try:
                    return str(int(val, 16))
                except ValueError:
                    return val
            return str(float(val))
        
        # Fill missing values and convert to string. 
        df_filled = df[columns].fillna('0').apply(
            lambda x: x.apply(convert_value)
        )
        return self.encoders[encoder_name].fit_transform(df_filled.values.reshape(-1, 1)) 
        # reshape(-1, 1) satisfies the input format of one-hot encoder. 
        # However, the original data may have several entries, like (2, 3). 
        # The final output whose shape is (-1, 1) may need to be reshaped again to match the original entry number, 
        # or the input need to be one feature per entry. 
    
    def _encode_ip(self, ip_series: pd.Series) -> np.ndarray:
        """
        Transform the IP address into dot decimal format, and then turn it into a normalized 4-dimensional numpy array. 
        """
        # Fill missing values
        ip_series = ip_series.fillna('00000000')
        # Convert hexadecimal IP address to dot decimal format
        def hex_to_dot_decimal(hex_ip):
            try:
                if isinstance(hex_ip, str) and hex_ip.isalnum():
                    # Convert hexadecimal string to integer
                    ip_int = int(hex_ip, 16)
                else:
                    # If it is a floating point number, directly use it
                    ip_int = int(float(hex_ip))
                # Convert to dot decimal format
                return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
            except:
                return "0.0.0.0"
        
        # Convert IP address format
        ip_series = ip_series.apply(hex_to_dot_decimal)
        ip_parts = ip_series.str.split('.')
        ip_matrix = np.array([list(map(int, parts)) for parts in ip_parts])
        return ip_matrix / 255.0
    
    def _encode_mac(self, mac_series: pd.Series) -> np.ndarray:
        """将MAC地址转换为数值向量"""
        # 处理缺失值
        mac_series = mac_series.fillna('000000000000')
        # 移除冒号并转换为数值
        mac_clean = mac_series.str.replace(':', '')
        def convert_mac(x):
            try:
                if isinstance(x, str) and x.isalnum():
                    return int(x, 16)
                return int(float(x))
            except:
                return 0
        # 修改这里，确保输出维度正确
        mac_values = np.array([list(map(convert_mac, mac_clean))])
        return mac_values.T / 255.0  # 转置以匹配其他特征的维度
    
    def _encode_hex_string(self, hex_series: pd.Series) -> np.ndarray:
        """将十六进制字符串转换为数值向量"""
        # 处理缺失值
        hex_series = hex_series.fillna('0')
        def convert_hex(x):
            try:
                if isinstance(x, str) and x.isalnum():
                    return int(x, 16)
                return int(float(x))
            except:
                return 0
        hex_values = hex_series.apply(convert_hex)
        return hex_values.values.reshape(-1, 1) / 65535.0
    
    def _encode_timestamp(self, timestamp_series: pd.Series) -> np.ndarray:
        """将时间戳转换为数值向量"""
        # 处理缺失值
        timestamp_series = timestamp_series.fillna(pd.Timestamp.now().strftime('%y%m%d%H%M%SZ'))
        
        def convert_timestamp(x):
            try:
                if pd.isna(x):
                    return pd.Timestamp.now().timestamp()
                
                # 尝试解析不同格式的时间戳
                if isinstance(x, str):
                    # 如果是十六进制格式
                    if x.isalnum():
                        try:
                            x = str(int(x, 16))
                        except ValueError:
                            pass
                    
                    # 尝试解析标准格式
                    try:
                        return pd.to_datetime(x, format='%y%m%d%H%M%SZ').timestamp()
                    except ValueError:
                        pass
                    
                    # 尝试解析ISO格式
                    try:
                        return pd.to_datetime(x).timestamp()
                    except ValueError:
                        pass
                
                # 如果是数值，假设是Unix时间戳
                try:
                    x = float(x)
                    # 如果数字太大，可能是纳秒级时间戳，转换为秒
                    if x > 1e12:
                        x = x / 1e9
                    return x
                except (ValueError, TypeError):
                    pass
                
                # 如果所有尝试都失败，返回当前时间
                return pd.Timestamp.now().timestamp()
                
            except Exception as e:
                print(f"Warning: Error converting timestamp {x}: {str(e)}")
                return pd.Timestamp.now().timestamp()
        
        # 转换时间戳
        timestamps = timestamp_series.apply(convert_timestamp)
        return timestamps.values.reshape(-1, 1)
    
    def _encode_certificate(self, cert_data: pd.DataFrame) -> np.ndarray:
        """处理证书相关字段"""
        def convert_value(x):
            try:
                if isinstance(x, str) and x.isalnum():
                    return int(x, 16)
                return int(float(x))
            except:
                return 0
        
        # 处理证书版本
        version = cert_data['tls.x509af.version'].fillna('0').apply(convert_value).values.reshape(-1, 1)
        
        # 处理序列号
        serial = cert_data['tls.x509af.serialNumber'].fillna('0').apply(convert_value).values.reshape(-1, 1)
        
        # 处理有效期
        not_before = self._encode_timestamp(cert_data['tls.x509af.notBefore'])
        not_after = self._encode_timestamp(cert_data['tls.x509af.notAfter'])
        
        # 处理密钥用法
        key_usage_cols = [
            'tls.x509ce.KeyUsage.digitalSignature',
            'tls.x509ce.KeyUsage.contentCommitment',
            'tls.x509ce.KeyUsage.keyEncipherment',
            'tls.x509ce.KeyUsage.dataEncipherment',
            'tls.x509ce.KeyUsage.keyAgreement',
            'tls.x509ce.KeyUsage.keyCertSign',
            'tls.x509ce.KeyUsage.cRLSign',
            'tls.x509ce.KeyUsage.encipherOnly',
            'tls.x509ce.KeyUsage.decipherOnly'
        ]
        key_usage = cert_data[key_usage_cols].fillna('0').apply(
            lambda x: x.apply(convert_value)
        ).values
        
        return np.concatenate([version, serial, not_before, not_after, key_usage], axis=1)
    
    def embed_features(self, df_dict: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """整合所有特征embedding"""
        features_list = []
        
        try:
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
            port_features = self._encode_categorical(
                base_df, ['tcp.srcport', 'tcp.dstport'], 'port')
            proto_features = self._encode_categorical(
                base_df, ['ip.proto'], 'proto')
            
            # 确保端口和协议特征的维度正确
            if port_features.shape[0] != base_df.shape[0]:
                port_features = port_features.reshape(base_df.shape[0], -1)
            if proto_features.shape[0] != base_df.shape[0]:
                proto_features = proto_features.reshape(base_df.shape[0], -1)
            
            features_list.append(port_features)
            features_list.append(proto_features)
            
            # 4. TCP选项
            if '2.csv' in df_dict:
                tcp_options_df = df_dict['2.csv']
                tcp_options_features = self._encode_hex_string(tcp_options_df['tcp.options'])
                if tcp_options_features.shape[0] != base_df.shape[0]:
                    tcp_options_features = np.zeros((base_df.shape[0], 1))
                features_list.append(tcp_options_features)
            
            # 5. TLS记录层
            if '5.csv' in df_dict:
                tls_record_df = df_dict['5.csv']
                tls_content_features = self._encode_categorical(
                    tls_record_df, ['tls.record.content_type'], 'tls_content_type')
                if tls_content_features.shape[0] != base_df.shape[0]:
                    tls_content_features = np.zeros((base_df.shape[0], tls_content_features.shape[1]))
                features_list.append(tls_content_features)
            
            if '6.csv' in df_dict:
                tls_version_df = df_dict['6.csv']
                tls_version_features = self._encode_categorical(
                    tls_version_df, ['tls.record.version'], 'tls_version')
                tls_length_features = self._encode_hex_string(tls_version_df['tls.record.length'])
                
                if tls_version_features.shape[0] != base_df.shape[0]:
                    tls_version_features = np.zeros((base_df.shape[0], tls_version_features.shape[1]))
                if tls_length_features.shape[0] != base_df.shape[0]:
                    tls_length_features = np.zeros((base_df.shape[0], 1))
                
                features_list.append(tls_version_features)
                features_list.append(tls_length_features)
            
            # 6. TLS握手层
            if '7.csv' in df_dict:
                tls_handshake_df = df_dict['7.csv']
                tls_handshake_features = self._encode_categorical(
                    tls_handshake_df, ['tls.handshake.type'], 'tls_handshake_type')
                tls_handshake_length = self._encode_hex_string(tls_handshake_df['tls.handshake.length'])
                
                if tls_handshake_features.shape[0] != base_df.shape[0]:
                    tls_handshake_features = np.zeros((base_df.shape[0], tls_handshake_features.shape[1]))
                if tls_handshake_length.shape[0] != base_df.shape[0]:
                    tls_handshake_length = np.zeros((base_df.shape[0], 1))
                
                features_list.append(tls_handshake_features)
                features_list.append(tls_handshake_length)
            
            # 7. TLS扩展
            if '8.csv' in df_dict:
                tls_ext_df = df_dict['8.csv']
                tls_ext_features = self._encode_categorical(
                    tls_ext_df, ['tls.handshake.extension.type'], 'tls_extension_type')
                if tls_ext_features.shape[0] != base_df.shape[0]:
                    tls_ext_features = np.zeros((base_df.shape[0], tls_ext_features.shape[1]))
                features_list.append(tls_ext_features)
            
            # 8. 证书信息
            if '16.csv' in df_dict:
                cert_features = self._encode_certificate(df_dict['16.csv'])
                if cert_features.shape[0] != base_df.shape[0]:
                    cert_features = np.zeros((base_df.shape[0], cert_features.shape[1]))
                features_list.append(cert_features)
            
            # 合并所有特征
            all_features = np.concatenate(features_list, axis=1)
            return torch.FloatTensor(all_features)
            
        except Exception as e:
            print(f"Error in embed_features: {str(e)}")
            print(f"Current features_list length: {len(features_list)}")
            for i, feat in enumerate(features_list):
                print(f"Feature {i} shape: {feat.shape}")
            raise

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
            df = pd.read_csv(file_path)
            # 确保所有列名都是字符串类型
            df.columns = df.columns.astype(str)
            df_dict[f"{i}.csv"] = df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
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