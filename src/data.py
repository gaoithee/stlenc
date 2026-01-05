import copy
import random
import re
from typing import List
from train import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from utils import simplify

# ==========================================
# 1. TRASFORMAZIONI TEMPORALI (STRING-BASED)
# ==========================================

def apply_temporal_complications(formula_str):
    """Manipola la stringa per split (L e R), dualità e Until ridondante."""
    
    # --- DUALITÀ: always[L,R] -> not ( eventually[L,R] ( not ... ) ) ---
    if "always[" in formula_str and random.random() < 0.4:
        formula_str = re.sub(r"always\[(\d+),(\d+|inf)\]\s*\((.*)\)", 
                             r"not ( eventually[\1,\2] ( not (\3) ) )", formula_str)
    
    # --- SPLIT AVANZATO: Op[L1+L2, R1+R2] -> Op[L1, R1] ( Op[L2, R2] ) ---
    match = re.search(r"(always|eventually)\[(\d+),(\d+)\]", formula_str)
    if match and random.random() < 0.5:
        tag, l_total_str, r_total_str = match.groups()
        L_total, R_total = int(l_total_str), int(r_total_str)
        
        if R_total > L_total + 2: # Spazio per uno split significativo
            # Splittiamo L_total = L1 + L2
            L1 = random.randint(0, L_total)
            L2 = L_total - L1
            # Splittiamo R_total = R1 + R2 (garantendo R1 > L1 e R2 > L2)
            R1 = random.randint(L1 + 1, R_total - L2 - 1)
            R2 = R_total - R1
            
            new_pattern = f"{tag}[{L1},{R1}] ( {tag}[{L2},{R2}]"
            formula_str = formula_str.replace(f"{tag}[{l_total_str},{r_total_str}]", new_pattern) + " )"

    # --- UNTIL RIDONDANTE: phi -> (phi until[L,R] phi) ---
    # Lo applichiamo solo se la formula non è già troppo complessa
    if random.random() < 0.2 and len(formula_str) < 100:
        # Estraiamo un pezzo di formula (o l'intera formula)
        formula_str = f"( ({formula_str}) until[0,10] ({formula_str}) )"
            
    return formula_str

# ==========================================
# 2. COMPLICAZIONE LOGICA (OBJECT-BASED)
# ==========================================

def complicate_logic(node):
    try:
        node_cp = copy.deepcopy(node)
        r = random.random()
        
        if r < 0.2: return Not(Not(node_cp))
        if r < 0.4: return And(node_cp, copy.deepcopy(node_cp)) if random.random() > 0.5 else Or(node_cp, copy.deepcopy(node_cp))
        
        # Negazione atomo (LTE <-> NOT GTE)
        if isinstance(node_cp, Atom) and r < 0.8:
            return Not(Atom(node_cp.var_index, node_cp.threshold, not node_cp.lte))
        
        if hasattr(node_cp, 'child'):
            node_cp.child = complicate_logic(node_cp.child)
        elif hasattr(node_cp, 'left_child'):
            node_cp.left_child = complicate_logic(node_cp.left_child)
            node_cp.right_child = complicate_logic(node_cp.right_child)
        return node_cp
    except: return node

# ==========================================
# 3. GENERATORE VARIANTI
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
        variants.add(to_string(phi_clean))
        
        attempts = 0
        while len(variants) < num_variants + 1 and attempts < 200:
            v_obj = complicate_logic(copy.deepcopy(phi_clean))
            v_str = to_string(v_obj)
            v_str = apply_temporal_complications(v_str)
            variants.add(v_str)
            attempts += 1
    except: return []
    return list(variants)
