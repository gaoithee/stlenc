import copy
import random
import re
from typing import List
from train_arch import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("saracandu/stlenc", trust_remote_code=True)

# ==========================================
# UTILS DI STRUTTURA
# ==========================================

def get_tree_depth(node):
    """Calcola la profondità reale dell'albero STL (altezza del nodo)."""
    if isinstance(node, Atom):
        return 1
    if hasattr(node, 'child'):
        return 1 + get_tree_depth(node.child)
    if hasattr(node, 'left_child'):
        return 1 + max(get_tree_depth(node.left_child), get_tree_depth(node.right_child))
    return 1

# ==========================================
# 1. TRASFORMAZIONI TEMPORALI (STRING-BASED)
# ==========================================

def apply_temporal_complications(formula_str):
    """Gestisce solo la dualità via Regex. Lo split ora è gestito dagli oggetti."""
    r = random.random()
    # --- DUALITÀ (Sintassi pura) ---
    if "always[" in formula_str and r < 0.4:
        formula_str = re.sub(r"always\[(\d+),(\d+|inf)\]\s*\((.*)\)", 
                             r"not ( eventually[\1,\2] ( not (\3) ) )", formula_str)
    elif "eventually[" in formula_str and r < 0.4:
        formula_str = re.sub(r"eventually\[(\d+),(\d+|inf)\]\s*\((.*)\)", 
                             r"not ( always[\1,\2] ( not (\3) ) )", formula_str)
    return formula_str

# ==========================================
# 2. COMPLICAZIONE LOGICA (OBJECT-BASED)
# ==========================================

def complicate_logic(node, max_overall_depth=25):
    """
    Versione ANTI-NOT: Impedisce catene triviali di negazioni e forza 
    la complessità su operatori temporali e binari.
    """
    if get_tree_depth(node) >= max_overall_depth:
        return copy.deepcopy(node)

    try:
        node_cp = copy.deepcopy(node)
        r = random.random()
        
        def next_step(n): 
            return complicate_logic(n, max_overall_depth)

        # 1. DOPPIA NEGAZIONE (Quasi azzerata: 0.1%)
        # Aggiunto controllo: se il nodo è già un Not, NON aggiungerne altri
        if r < 0.001 and not isinstance(node_cp, Not):
            return Not(Not(next_step(node_cp)))

        # 2. DE MORGAN (Bilanciato al 10%)
        elif isinstance(node_cp, (And, Or)) and r < 0.10:
            if isinstance(node_cp, And):
                return Not(Or(Not(next_step(node_cp.left_child)), Not(next_step(node_cp.right_child))))
            else:
                return Not(And(Not(next_step(node_cp.left_child)), Not(next_step(node_cp.right_child))))

        # 3. COMPOSIZIONE / SOMMA BOUND (AUMENTATA al 35% -> r < 0.45)
        # Forza la profondità tramite nidificazione temporale seria
        elif isinstance(node_cp, (Globally, Eventually)) and r < 0.45:
            L_tot, R_tot = int(node_cp.left_time_bound), node_cp.right_time_bound
            if not node_cp.right_unbound and R_tot > L_tot + 1:
                split = random.uniform(0.3, 0.7)
                L1, R1 = int(L_tot * split), int(R_tot * split)
                L2, R2 = L_tot - L1, R_tot - R1
                if R2 <= L2: R2 = L2 + 1
                inner = type(node_cp)(next_step(node_cp.child), L2, R2, False)
                return type(node_cp)(inner, L1, R1, False)

        # 4. UNTIL COMPLESSO (AUMENTATA al 25% -> r < 0.70)
        # L'Until è l'operatore più ricco per la profondità semantica
        elif r < 0.70 and not isinstance(node_cp, (Until, Atom)):
            return Until(next_step(node_cp), copy.deepcopy(node_cp), 0, random.randint(1, 10), False)

        # 5. IDENTITÀ TEMPORALE / SHIFT PURO (Ridotto al 5% -> r < 0.75)
        elif isinstance(node_cp, (Globally, Eventually)) and r < 0.75:
            return type(node_cp)(next_step(node_cp), 0, 0, False)

        # 6. DISTRIBUTIVITÀ (AUMENTATA al 15% -> r < 0.90)
        elif r < 0.90:
            if isinstance(node_cp, Globally) and isinstance(node_cp.child, And):
                L, R, U = node_cp.left_time_bound, node_cp.right_time_bound, node_cp.right_unbound
                return And(Globally(next_step(node_cp.child.left_child), L, R, U), 
                           Globally(next_step(node_cp.child.right_child), L, R, U))
            elif isinstance(node_cp, Eventually) and isinstance(node_cp.child, Or):
                L, R, U = node_cp.left_time_bound, node_cp.right_time_bound, node_cp.right_unbound
                return Or(Eventually(next_step(node_cp.child.left_child), L, R, U), 
                          Eventually(next_step(node_cp.child.right_child), L, R, U))

        # 7. NEGAZIONE PREDICATO (Solo per Atomi, r < 0.98)
        elif isinstance(node_cp, Atom) and r < 0.98:
            # Evitiamo di negare se siamo già dentro un Not (per non fare catene)
            return Not(Atom(node_cp.var_index, node_cp.threshold, not node_cp.lte))

        # Default: Discesa ricorsiva senza aggiungere operatori
        else:
            if hasattr(node_cp, 'child'): node_cp.child = next_step(node_cp.child)
            elif hasattr(node_cp, 'left_child'):
                node_cp.left_child = next_step(node_cp.left_child)
                node_cp.right_child = next_step(node_cp.right_child)
            return node_cp
            
    except:
        return node
     
# ==========================================
# 3. GENERAZIONE E FILTRAGGIO
# ==========================================

def to_string(node) -> str:
    if isinstance(node, Atom):
        val = "{:.4f}".format(node.threshold)
        if val == "-0.0000": val = "0.0000"
        return f"x_{node.var_index} {'<=' if node.lte else '>='} {val}"
    if isinstance(node, Not): return f"not ( {to_string(node.child)} )"
    if isinstance(node, (And, Or)):
        op = "and" if isinstance(node, And) else "or"
        return f"( {to_string(node.left_child)} {op} {to_string(node.right_child)} )"
    if isinstance(node, (Eventually, Globally, Until)):
        tag = "eventually" if isinstance(node, Eventually) else "always" if isinstance(node, Globally) else "until"
        r_str = "inf" if node.right_unbound else str(int(node.right_time_bound))
        b = f"[{int(node.left_time_bound)},{r_str}]"
        if isinstance(node, Until): return f"( {to_string(node.left_child)} until{b} {to_string(node.right_child)} )"
        return f"{tag}{b} ( {to_string(node.child)} )"
    return str(node)

def apply_parametric_shift(node):
    """Insegna sia la stabilità locale (vibrazione) che la linearità globale (shift)."""
    node_cp = copy.deepcopy(node)
    
    def get_new_val_combined(val, is_time=False):
        # 50% vibrazione (moltiplicativa), 50% shift (additiva)
        use_vibration = random.random() < 0.5
        
        if is_time:
            if use_vibration:
                scale = random.uniform(0.01, 0.15)
                modifier = 1 + scale if random.random() > 0.5 else max(0.0, 1 - scale)
                new_val = int(val * modifier)
            else:
                delta = random.randint(-15, 40)
                new_val = int(val + delta)
            return max(0, new_val)
        else:
            if use_vibration:
                scale = random.uniform(0.01, 0.10)
                modifier = 1 + scale if random.random() > 0.5 else 1 - scale
                new_val = val * modifier
            else:
                delta = random.uniform(-6.0, 6.0)
                new_val = val + delta
            # Formattazione coerente con il tuo to_string
            return "{:.4f}".format(new_val).replace("-0.0000", "0.0000")

    if isinstance(node_cp, Atom):
        node_cp.threshold = float(get_new_val_combined(node_cp.threshold, is_time=False))
    elif isinstance(node_cp, (Globally, Eventually, Until)):
        # Salviamo i bound originali per gestire la coerenza R > L
        old_L = node_cp.left_time_bound
        old_R = node_cp.right_time_bound if not node_cp.right_unbound else None
        
        # Applichiamo lo shift/vibrazione a L
        node_cp.left_time_bound = get_new_val_combined(old_L, is_time=True)
        
        if not node_cp.right_unbound:
            # Calcoliamo una nuova ampiezza dell'intervallo (W) basata su quella vecchia
            old_width = max(1, old_R - old_L)
            # Facciamo vibrare anche l'ampiezza
            new_width = max(1, int(old_width * random.uniform(0.6, 1.8)))
            node_cp.right_time_bound = node_cp.left_time_bound + new_width
    
    # Ricorsione
    if hasattr(node_cp, 'child'): 
        node_cp.child = apply_parametric_shift(node_cp.child)
    elif hasattr(node_cp, 'left_child'):
        node_cp.left_child = apply_parametric_shift(node_cp.left_child)
        node_cp.right_child = apply_parametric_shift(node_cp.right_child)
    return node_cp

def generate_stratified_variants(raw_formula_str, num_variants=30, global_max_depth=20, max_tokens=500, min_depth=5):
    stratified_rows = []
    seen_variants = {raw_formula_str} 
    
    try:
        gold_node = from_string_to_formula(raw_formula_str)
        gold_depth = get_tree_depth(gold_node)
    except Exception:
        return []

    # Calcoliamo la profondità target: se la gold è già profonda, usiamo quella, 
    # altrimenti puntiamo almeno a min_depth.
    target_min_depth = max(min_depth, gold_depth)

    categories = [
        ("identity", max(1, int(num_variants * 0.05))), 
        ("logic_exact", max(1, int(num_variants * 0.25))), 
        ("param_shift_simple", max(1, int(num_variants * 0.30))), 
        ("hybrid_complex", num_variants - (max(1, int(num_variants * 0.05)) + max(1, int(num_variants * 0.25)) + max(1, int(num_variants * 0.30))))
    ]

    for cat_name, count in categories:
        generated = 0
        attempts = 0
        while generated < count and attempts < 400: # Aumentiamo gli attempts: la selezione è dura
            attempts += 1
            v_node = copy.deepcopy(gold_node)
            
            # Applichiamo complicazione logica più volte se necessario per raggiungere la profondità
            # specialmente per le categorie 'logic_exact' e 'hybrid'
            if cat_name in ["logic_exact", "hybrid_complex"]:
                current_v_depth = get_tree_depth(v_node)
                # Loop di forzatura: se è troppo semplice, complica ancora
                while current_v_depth < target_min_depth and current_v_depth < global_max_depth:
                    v_node = complicate_logic(v_node, max_overall_depth=global_max_depth)
                    new_depth = get_tree_depth(v_node)
                    if new_depth <= current_v_depth: # Evita loop infiniti se non riesce a complicare
                        break
                    current_v_depth = new_depth

            if cat_name in ["param_shift_simple", "hybrid_complex"]:
                v_node = apply_parametric_shift(v_node)
            
            v_str = to_string(v_node)
            
            if cat_name in ["logic_exact", "hybrid_complex"]:
                v_str = apply_temporal_complications(v_str)

            if v_str in seen_variants:
                continue

            try:
                parsed_back = from_string_to_formula(v_str)
                d_final = get_tree_depth(parsed_back)
                
                # --- IL FILTRO DI FERRO ---
                # Per le categorie non-parametriche (che devono essere complesse), 
                # scartiamo se non raggiungono la profondità minima.
                if cat_name in ["logic_exact", "hybrid_complex"] and d_final < target_min_depth:
                    continue
                
                if d_final > global_max_depth: continue
                if len(tokenizer.encode(v_str, add_special_tokens=False)) > max_tokens: continue
                
                stratified_rows.append({
                    "type": cat_name,
                    "variant": v_str,
                    "label": 1.0 if "exact" in cat_name or "ident" in cat_name else 0.0,
                    "depth": d_final
                })
                seen_variants.add(v_str)
                generated += 1
            except:
                continue
            
    return stratified_rows

