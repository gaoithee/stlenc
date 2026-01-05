import copy
import random
import re
from typing import List
from train_new import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from utils import simplify

# ==========================================
# 1. TRASFORMAZIONI TEMPORALI (STRING-BASED)
# ==========================================

def apply_temporal_complications(formula_str):
    """Manipola la stringa per forzare split e dualità senza crashare gli oggetti."""
    
    # --- DUALITÀ: always[L,R] -> not ( eventually[L,R] ( not ... ) ) ---
    if "always[" in formula_str and random.random() < 0.5:
        formula_str = re.sub(r"always\[(\d+),(\d+|inf)\]\s*\((.*)\)", 
                             r"not ( eventually[\1,\2] ( not (\3) ) )", formula_str)
    
    # --- SPLIT: always[L,R] -> always[L,M] ( always[0,R-M] ) ---
    # Cerchiamo un bound numerico da splittare
    match = re.search(r"(always|eventually)\[(\d+),(\d+)\]", formula_str)
    if match and random.random() < 0.5:
        tag, l_val, r_val = match.groups()
        L, R = int(l_val), int(r_val)
        if R > L + 1:
            mid = random.randint(L + 1, R - 1)
            diff = R - mid
            new_pattern = f"{tag}[{L},{mid}] ( {tag}[0,{diff}]"
            formula_str = formula_str.replace(f"{tag}[{l_val},{r_val}]", new_pattern) + " )"
            
    return formula_str

# ==========================================
# 2. COMPLICAZIONE LOGICA (OBJECT-BASED)
# ==========================================

def complicate_logic(node):
    """Applica trasformazioni booleane standard sugli oggetti."""
    try:
        node_cp = copy.deepcopy(node)
        r = random.random()
        
        if r < 0.3: # Doppia negazione
            return Not(Not(node_cp))
        if r < 0.6: # Idempotenza
            return And(node_cp, copy.deepcopy(node_cp)) if random.random() > 0.5 else Or(node_cp, copy.deepcopy(node_cp))
        if isinstance(node_cp, Atom) and r < 0.9: # Negazione atomo
            return Not(Atom(node_cp.var_index, node_cp.threshold, not node_cp.lte))
        
        # Ricorsione semplice
        if hasattr(node_cp, 'child'):
            node_cp.child = complicate_logic(node_cp.child)
        elif hasattr(node_cp, 'left_child'):
            node_cp.left_child = complicate_logic(node_cp.left_child)
            node_cp.right_child = complicate_logic(node_cp.right_child)
            
        return node_cp
    except:
        return node

# ==========================================
# 3. GENERATORE UNIFICATO
# ==========================================

def to_string(node) -> str:
    if isinstance(node, Atom):
        return f"x_{node.var_index} {'<=' if node.lte else '>='} {node.threshold:.4f}"
    if isinstance(node, Not):
        return f"not ( {to_string(node.child)} )"
    if isinstance(node, (And, Or)):
        op = "and" if isinstance(node, And) else "or"
        return f"( {to_string(node.left_child)} {op} {to_string(node.right_child)} )"
    if isinstance(node, (Eventually, Globally, Until)):
        tag = "eventually" if isinstance(node, Eventually) else "always" if isinstance(node, Globally) else "until"
        r_str = "inf" if node.right_unbound else str(int(node.right_time_bound))
        b = f"[{int(node.left_time_bound)},{r_str}]"
        if isinstance(node, Until):
            return f"( {to_string(node.left_child)} until{b} {to_string(node.right_child)} )"
        return f"{tag}{b} ( {to_string(node.child)} )"
    return str(node)

def generate_variants(raw_formula: str, num_variants: int = 5) -> List[str]:
    variants = set()
    try:
        phi = from_string_to_formula(raw_formula)
        phi_clean = simplify(phi)
        
        # Target canonico
        target = to_string(phi_clean)
        variants.add(target)
        
        attempts = 0
        while len(variants) < num_variants + 1 and attempts < 100:
            # 1. Complicazione Logica
            v_obj = complicate_logic(copy.deepcopy(phi_clean))
            v_str = to_string(v_obj)
            
            # 2. Complicazione Temporale (Sulla stringa prodotta)
            v_str = apply_temporal_complications(v_str)
            
            variants.add(v_str)
            attempts += 1
            
    except Exception as e:
        return []
    return list(variants)
