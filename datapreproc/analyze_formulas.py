import pandas as pd
import re
import networkx as nx
from tqdm import tqdm
from typing import List
from datasets import load_dataset
from train_arch import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from transformers import AutoTokenizer

# 1. Inizializzazione
# Uso saracandu/stlenc-arch se disponibile, o bert-base-uncased come fallback per il conteggio token
try:
    tokenizer = AutoTokenizer.from_pretrained("saracandu/stlenc-arch", trust_remote_code=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ==========================================
# FUNZIONI DI SUPPORTO PER GRAFO E METRICHE
# ==========================================
def get_name_given_type(formula):
    name_dict = {And: 'and', Or: 'or', Not: 'not', Eventually: 'F', Globally: 'G', Until: 'U', Atom: 'x'}
    return name_dict[type(formula)]

def get_id(child_name, name, label_dict, idx):
    while child_name in label_dict.keys():
        idx += 1
        child_name = name + "(" + str(idx) + ")"
    return child_name, idx

def add_internal_child(current_child, current_idx, label_dict):
    name = get_name_given_type(current_child)
    child_name = f"{name}({current_idx})"
    child_name, current_idx = get_id(child_name, name, label_dict, current_idx)
    return child_name, current_idx

def traverse_formula(formula, idx, label_dict):
    """
    Versione corretta e semplificata per la costruzione del grafo
    """
    edges = []
    current_name = f"{get_name_given_type(formula)}({idx})"
    label_dict[current_name] = True # Segnaposto per esistenza nodo

    # Operatori Binari
    if isinstance(formula, (And, Or, Until)):
        # Left
        l_name, next_idx = add_internal_child(formula.left_child, idx + 1, label_dict)
        edges.append([current_name, l_name])
        e, d = traverse_formula(formula.left_child, next_idx, label_dict)
        edges += e
        # Right
        r_name, next_idx = add_internal_child(formula.right_child, next_idx + 1, label_dict)
        edges.append([current_name, r_name])
        e, d = traverse_formula(formula.right_child, next_idx, label_dict)
        edges += e
    # Operatori Unari
    elif isinstance(formula, (Not, Eventually, Globally)):
        c_name, next_idx = add_internal_child(formula.child, idx + 1, label_dict)
        edges.append([current_name, c_name])
        e, d = traverse_formula(formula.child, next_idx, label_dict)
        edges += e
    
    return edges, label_dict

def get_metrics(f_str):
    """
    Estrae profondità, conteggio operatori e complessità temporale
    """
    node = from_string_to_formula(f_str)
    edges, labels = traverse_formula(node, 0, {})
    
    if not edges: # Caso formula atomica
        return 1, 0, 0
        
    graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
    depth = len(nx.dag_longest_path(graph)) - 1
    
    # Conteggio operatori temporali (F, G, U) vs Logici
    temp_ops = len(re.findall(r'(always|eventually|until|globally|finally|U|G|F)', f_str, re.IGNORECASE))
    logic_ops = len(re.findall(r'(and|or|not|&&|\|\||!)', f_str, re.IGNORECASE))
    
    return depth, temp_ops, logic_ops

# ==========================================
# 2. ANALISI DATASET
# ==========================================
def analyze_dataset(formula_list: List[str]):
    stats = []
    print(f"Analisi in corso su {len(formula_list)} formule...")
    
    for f_str in tqdm(formula_list):
        try:
            depth, t_ops, l_ops = get_metrics(f_str)
            tokens = tokenizer.encode(f_str, add_special_tokens=False)
            
            stats.append({
                "depth": depth,
                "temp_ops": t_ops,
                "logic_ops": l_ops,
                "total_ops": t_ops + l_ops,
                "token_len": len(tokens),
                "chars_len": len(f_str)
            })
        except Exception as e:
            continue
            
    return pd.DataFrame(stats)

# 3. Caricamento e Run
print("Caricamento dataset...")
dataset = load_dataset("saracandu/stl_updated", split="train")
col = "formula" if "formula" in dataset.column_names else dataset.column_names[0]

df = analyze_dataset(dataset[col])

# 4. Visualizzazione Risultati
if not df.empty:
    print("\n" + "="*50)
    print("COMPOSIZIONE DATASET STL_NEW")
    print("="*50)
    
    # Statistiche Descrittive
    summary = df.describe().loc[['mean', 'min', 'max', '50%']]
    summary.index = ['Media', 'Min', 'Max', 'Mediana']
    print(summary.T)
    
    print("\n--- Distribuzione Profondità (Top 5) ---")
    print(df['depth'].value_counts(normalize=True).head(5).mul(100).round(2).astype(str) + '%')

    print("\n--- Correlazione Token/Operatori ---")
    corr = df['token_len'].corr(df['total_ops'])
    print(f"Pearson Correlation (Tokens vs Ops): {corr:.4f}")

    # Analisi della complessità temporale
    avg_temp = df['temp_ops'].mean()
    print(f"\nMedia Operatori Temporali per formula: {avg_temp:.2f}")
else:
    print("Errore: Nessun dato analizzato.")