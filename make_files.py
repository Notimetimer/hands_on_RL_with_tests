import os
import re

with open("MAPPO_MPE训练2_带AT与id相加.py", "r", encoding="utf-8") as f:
    f2 = f.read()

with open("MAPPO_MPE训练3_带GAT使用MHA与padding.py", "r", encoding="utf-8") as f:
    f3 = f.read()

def get_class(text, class_name):
    # Matches starting from "class Name" up to the next "class " or "# ==="
    pattern = r'(class ' + class_name + r'\(.*?(?=\nclass |\n# ===|\Z))'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print(f"Could not find {class_name}")
    return None

c_mha_critic = get_class(f2, "UnifiedAttentionValueNet")
c_gat_critic = get_class(f3, "StructuredGATCritic")

gat_dep1 = get_class(f3, "OfficialMHALayerWithMask")
gat_dep2 = get_class(f3, "QKVLayerWithEdgeMask")
gat_dep3 = get_class(f3, "GATBlock")
gat_deps = gat_dep1 + "\n" + gat_dep2 + "\n" + gat_dep3 + "\n"

# -------------
# MAKE FILE 4 (MHA Actor + GAT Critic)
# Based on File 2
f4 = f2.replace(c_mha_critic, gat_deps + c_gat_critic)
# Replace MAPPO critic init
f4 = re.sub(
    r'self\.critic = UnifiedAttentionValueNet\([^)]+\)\.to\(device\)',
    'self.critic = StructuredGATCritic(n_agents=num_agents, node_dim=obs_dim, n_landmarks=num_agents, hidden_dim=critic_hidden).to(device)',
    f4
)
with open("MAPPO_MPE训练4_MHA_Actor搭配GAT_Critic.py", "w", encoding="utf-8") as f:
    f.write(f4)

# -------------
# MAKE FILE 5 (GAT Actor + MHA Critic)
# Based on File 3
f5 = f3.replace(c_gat_critic, c_mha_critic)
# Replace MAPPO critic init
f5 = re.sub(
    r'self\.critic = StructuredGATCritic\(.*?\)\.to\(device\)',
    'self.critic = UnifiedAttentionValueNet(num_agents, node_dim=obs_dim, hidden_dim=critic_hidden).to(device)',
    f5,
    flags=re.DOTALL
)
with open("MAPPO_MPE训练5_GAT_Actor搭配MHA_Critic.py", "w", encoding="utf-8") as f:
    f.write(f5)

print("Created File 4 and File 5 successfully.")
