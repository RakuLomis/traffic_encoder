from torch_geometric.loader import DataLoader
import torch

def diagnose_gate_weights_for_class(model, dataset, target_class_name, label_to_int, device):
    """
    为一个特定的目标类别，诊断其相关特征的门控权重。
    """
    print("\n" + "-"*30)
    print(f"Diagnosing for class: '{target_class_name}'")
    print("-"*30)

    # 1. 从数据集中找到一个属于该类别的样本
    target_label_id = label_to_int.get(target_class_name)
    if target_label_id is None:
        print(f" -> 错误: 类别 '{target_class_name}' 不在标签映射中。")
        return

    target_graph = None
    # 创建一个临时的DataLoader来高效地查找样本
    diag_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for g in diag_loader:
        if g.y.item() == target_label_id:
            target_graph = g
            break
    
    if target_graph is None:
        print(f" -> 警告: 在数据集中找不到任何 '{target_class_name}' 的样本，跳过诊断。")
        return

    # 2. 识别出这个样本实际拥有的特征（节点）
    #    data.keys 是PyG中获取一个Data对象所有属性的正确方法
    # present_fields_in_sample = [f for f in model.node_fields if f in target_graph.keys]
    present_fields_in_sample = [f for f in model.node_fields if f in target_graph.keys()]
    
    if not present_fields_in_sample:
        print(" -> 警告: 这个样本不包含任何模型已知的特征。")
        return
        
    print(f" -> 样本实际用到的节点字段: {present_fields_in_sample}")
    print(" -> 正在检查这些字段的门控权重...")

    # 3. 检查这些字段对应的门控权重
    with torch.no_grad():
        g = torch.sigmoid(model.feature_mask_logits).cpu()
        for field_name in present_fields_in_sample:
            try:
                # 从模型的总节点列表中找到该特征的索引
                idx = model.node_fields.index(field_name)
                gate_value = g[idx].item()
                
                # 为了突出显示，我们格式化输出
                status = "OPEN" if gate_value > 0.5 else "CLOSED"
                if gate_value < 0.1: status = "!!! ALMOST KILLED !!!"

                print(f'   - {field_name:35s} gate = {gate_value:.4f}  ({status})')
            except ValueError:
                # 理论上不应发生，因为present_fields_in_sample已经是model.node_fields的子集
                pass

# ==================================================================================
