import copy
from collections import deque
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. PARSER CON FIX PER PARENTESI RIDONDANTI
# ==========================================

def set_time_thresholds(st):
    unbound, right_unbound = [True, False]
    l, r = [0, 0]
    if st[-1] == ']':
        unbound = False
        time_thresholds = st[st.index('[')+1:-1].split(",")
        l = int(time_thresholds[0])
        if time_thresholds[1] == 'inf': right_unbound = True
        else: r = int(time_thresholds[1]) - 1
    return unbound, right_unbound, l, r

def from_string_to_formula(st):
    st = st.strip()
    
    # --- NUOVO FIX: Sbucciatore di parentesi (Peeling) ---
    # Rimuove ( ( ... ) ) solo se avvolgono l'intera formula in modo bilanciato
    while st.startswith('(') and st.endswith(')'):
        inner = st[1:-1].strip()
        cnt, balanced = 0, True
        for char in inner:
            if char == '(': cnt += 1
            elif char == ')': cnt -= 1
            if cnt < 0: # Le parentesi non sono una coppia esterna (es: (A) or (B))
                balanced = False
                break
        if balanced and cnt == 0:
            st = inner
        else:
            break
    # -----------------------------------------------------

    st_split = st.split()
    if not st_split: return "Empty"

    # Ora root_arity sarà 1 per i casi "not (..." invece di 2
    root_arity = 2 if st.startswith('(') else 1
    
    if root_arity <= 1:
        root_op_str = copy.deepcopy(st_split[0])
        if root_op_str.startswith('x'):
            return "Atom"
        
        # Per gli operatori unari, puliamo la stringa figlia
        current_st = ' '.join(st_split[1:]).strip()
        # Se dopo l'operatore c'è un blocco parentesizzato, lo puliamo per la ricorsione
        if current_st.startswith('(') and current_st.endswith(')'):
             # Il peeling ricorsivo avverrà alla chiamata successiva
             pass
             
        if root_op_str == 'not': 
            return from_string_to_formula(current_st)
        
        try:
            un, run, l, r = set_time_thresholds(root_op_str)
            return from_string_to_formula(current_st)
        except:
            return "Error in thresholds"
    else:
        # Logica originale per And, Or, Until
        current_st = st_split[1:-1]
        if '(' in current_st:
            par_queue, par_idx_list = deque(), []
            for i, sub in enumerate(current_st):
                if sub == '(': par_queue.append(i)
                elif sub == ')': par_idx_list.append(tuple([par_queue.pop(), i]))
            children_range = []
            for begin, end in sorted(par_idx_list):
                if children_range and children_range[-1][1] >= begin - 1: 
                    children_range[-1][1] = max(children_range[-1][1], end)
                else: 
                    children_range.append([begin, end])
            
            if len(children_range) == 1:
                var_child_idx = 1 if children_range[0][0] <= 1 else 0
                if children_range[0][0] != 0 and current_st[children_range[0][0]-1][0:2] in ['no', 'ev', 'al']: 
                    children_range[0][0] -= 1
                
                # IndexError avveniva qui perché cercava l'operatore in una formula non-binaria
                op_idx = children_range[0][1]+1 if var_child_idx == 1 else children_range[0][0]-1
                if op_idx < 0 or op_idx >= len(current_st):
                    raise IndexError("Possibile formula unaria mal interpretata come binaria")
                
                op_str = current_st[op_idx]
            else:
                op_str = current_st[children_range[0][1]+1]
        else: 
            op_str = current_st[3]
        
        return "Parsed Binary"

# ==========================================
# 2. RUN DIAGNOSTIC
# ==========================================

def run_diagnostic(repo_id, num_samples=50000):
    print(f"📥 Caricamento dataset {repo_id}...")
    ds = load_dataset(repo_id, split="train")
    subset = ds.select(range(min(num_samples, len(ds))))
    
    crashes = 0
    examples = []

    print(f"🔬 Analisi di {len(subset)} varianti con Peeling Fix...")
    for row in tqdm(subset):
        formula = row["formula_variant"]
        try:
            from_string_to_formula(formula)
        except Exception as e:
            crashes += 1
            if len(examples) < 5:
                examples.append((formula, str(e)))

    rate = (crashes / len(subset)) * 100
    print(f"\nNuovo Crash Rate: {rate:.4f}% ({crashes}/{len(subset)})")
    if examples:
        print("\nCasi ancora problematici:")
        for f, err in examples:
            print(f"Errore: {err}\nFormula: {f}\n")

if __name__ == "__main__":
    run_diagnostic("saracandu/stl_formulae_variants")
