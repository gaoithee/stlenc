import copy
import random
from typing import List
from train_temp3 import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula

MAX_DEPTH = 25 

# def simplify(formula, depth = 0):
#     """
#     Recursively simplifies STL formulae using logical equivalences and temporal rules.
#     """
#     if depth > MAX_DEPTH:
#         return formula
#     try: node = copy.deepcopy(formula)
#     except: return formula
#     # --- Recursively simplify children first ---
#     if hasattr(node, "child"):
#         node.child = simplify(node.child, depth + 1)
#     elif hasattr(node, "left_child") and hasattr(node, "right_child"):
#         node.left_child = simplify(node.left_child, depth + 1)
#         node.right_child = simplify(node.right_child, depth + 1)

#     # --- NOT operator simplifications ---
#     if isinstance(node, Not):
#         child = node.child
#         if isinstance(child, Not):
#             return simplify(child.child, depth + 1)  # Double negation: ¬(¬φ) → φ
#         elif isinstance(child, Atom):
#             return Atom(  # Negated predicate
#                 var_index=child.var_index,
#                 threshold=copy.deepcopy(child.threshold),
#                 lte=not child.lte,
#             )
#         elif isinstance(child, Eventually):
#             return simplify(
#                 Globally(  # ¬F_I(φ) → G_I(¬φ)
#                     unbound=child.unbound,
#                     right_unbound=child.right_unbound,
#                     left_time_bound=child.left_time_bound,
#                     right_time_bound=child.right_time_bound,
#                     child=Not(child.child),
#                 ), depth + 1
#             )
#         elif isinstance(child, Globally):
#             return simplify(
#                 Eventually(  # ¬G_I(φ) → F_I(¬φ)
#                     unbound=child.unbound,
#                     right_unbound=child.right_unbound,
#                     left_time_bound=child.left_time_bound,
#                     right_time_bound=child.right_time_bound,
#                     child=Not(child.child),
#                 ), depth + 1
#             )
#         elif isinstance(child, Or):
#             # ¬(a ∨ b) → ¬a ∧ ¬b
#             return simplify(
#                 And(
#                     left_child=simplify(Not(child.left_child), depth + 1),
#                     right_child=simplify(Not(child.right_child), depth + 1),
#                 ), depth + 1
#             )
#         elif isinstance(child, And):
#             # ¬(a ∧ b) → ¬a ∨ ¬b
#             return simplify(
#                 Or(
#                     left_child=simplify(Not(child.left_child), depth + 1),
#                     right_child=simplify(Not(child.right_child), depth + 1),
#                 ), depth + 1
#             )
#     # --- AND simplifications ---
#     if isinstance(node, And):
#         l, r = node.left_child, node.right_child
#         if l == r:
#             return simplify(l, depth + 1)  # a ∧ a → a

#         # a ∧ (a ∨ b) → a
#         if isinstance(r, Or) and (l == r.left_child or l == r.right_child):
#             return simplify(l, depth + 1)
#         if isinstance(l, Or) and (r == l.left_child or r == l.right_child):
#             return simplify(r, depth + 1)

#         # a ∧ (a ∧ b) → a ∧ b
#         if isinstance(r, And):
#             if l == r.left_child:
#                 return simplify(And(left_child=l, right_child=r.right_child), depth + 1)
#             if l == r.right_child:
#                 return simplify(And(left_child=l, right_child=r.left_child), depth + 1)
#         if isinstance(l, And):
#             if r == l.left_child:
#                 return simplify(And(left_child=r, right_child=l.right_child), depth + 1)
#             if r == l.right_child:
#                 return simplify(And(left_child=r, right_child=l.left_child), depth + 1)

#     # --- OR simplifications ---
#     if isinstance(node, Or):
#         l, r = node.left_child, node.right_child
#         if l == r:
#             return simplify(l, depth + 1)  # a ∨ a → a

#         # a ∨ (a ∧ b) → a
#         if isinstance(r, And) and (l == r.left_child or l == r.right_child):
#             return simplify(l, depth + 1)
#         if isinstance(l, And) and (r == l.left_child or r == l.right_child):
#             return simplify(r, depth + 1)

#         # a ∨ (a ∨ b) → a ∨ b
#         if isinstance(r, Or):
#             if l == r.left_child:
#                 return simplify(Or(left_child=l, right_child=r.right_child), depth + 1)
#             if l == r.right_child:
#                 return simplify(Or(left_child=l, right_child=r.left_child), depth + 1)
#         if isinstance(l, Or):
#             if r == l.left_child:
#                 return simplify(Or(left_child=r, right_child=l.right_child), depth + 1)
#             if r == l.right_child:
#                 return simplify(Or(left_child=r, right_child=l.left_child), depth + 1)

#     # --- Nested Globally simplification: G_I(G_J(φ)) → G_{I+J}(φ) ---
#     if isinstance(node, Globally) and isinstance(node.child, Globally):
#         inner = node.child
#         child = inner.child
#         return simplify(
#             Globally(
#                 unbound=node.unbound or inner.unbound,
#                 right_unbound=node.right_unbound or inner.right_unbound,
#                 left_time_bound=node.left_time_bound + inner.left_time_bound,
#                 right_time_bound=(
#                     1
#                     if (node.right_unbound or inner.right_unbound)
#                     else node.right_time_bound + inner.right_time_bound - 1
#                 ),
#                 child=copy.deepcopy(child),
#             ), depth + 1
#         )
#     # --- Nested Eventually simplification: F_I(F_J(φ)) → F_{I+J}(φ) ---
#     if isinstance(node, Eventually) and isinstance(node.child, Eventually):
#         inner = node.child
#         child = inner.child
#         return simplify(
#             Eventually(
#                 unbound=node.unbound or inner.unbound,
#                 right_unbound=node.right_unbound or inner.right_unbound,
#                 left_time_bound=node.left_time_bound + inner.left_time_bound,
#                 right_time_bound=(
#                     1
#                     if (node.right_unbound or inner.right_unbound)
#                     else node.right_time_bound + inner.right_time_bound - 1
#                 ),
#                 child=copy.deepcopy(child),
#             ), depth + 1
#         )
#     # --- Until simplification: φ U φ → φ ---
#     if isinstance(node, Until) and node.left_child == node.right_child:
#         return simplify(node.left_child, depth + 1)

#     return node
def simplify(node, depth=0):
    if depth > MAX_DEPTH:
        return node

    # --- 1. SEMPLIFICAZIONE BOTTOM-UP (Figli prima dei Padri) ---
    if isinstance(node, Not):
        node.child = simplify(node.child, depth + 1)
    elif hasattr(node, "left_child") and hasattr(node, "right_child"):
        node.left_child = simplify(node.left_child, depth + 1)
        node.right_child = simplify(node.right_child, depth + 1)
    elif hasattr(node, "child"): # Per Globally/Eventually
        node.child = simplify(node.child, depth + 1)

    # --- 2. LOGICA DI TRASFORMAZIONE ---

    # NOT logic: De Morgan & Inversion
    if isinstance(node, Not):
        c = node.child
        if isinstance(c, Not): return c.child
        if isinstance(c, Atom): return Atom(c.var_index, c.threshold, not c.lte)
        if isinstance(c, Or): 
            return And(simplify(Not(c.left_child), depth+1), simplify(Not(c.right_child), depth+1))
        if isinstance(c, And): 
            return Or(simplify(Not(c.left_child), depth+1), simplify(Not(c.right_child), depth+1))
        if isinstance(c, Eventually):
            return Globally(c.left_time_bound, c.right_time_bound, c.right_unbound, simplify(Not(c.child), depth+1))
        if isinstance(c, Globally):
            return Eventually(c.left_time_bound, c.right_time_bound, c.right_unbound, simplify(Not(c.child), depth+1))

    # AND logic: Idempotenza & Assorbimento (Usa str() per il confronto sicuro)
    if isinstance(node, And):
        l_str, r_str = str(node.left_child), str(node.right_child)
        if l_str == r_str: return node.left_child
        if isinstance(node.right_child, Or):
            if l_str == str(node.right_child.left_child) or l_str == str(node.right_child.right_child):
                return node.left_child
        if isinstance(node.left_child, Or):
            if r_str == str(node.left_child.left_child) or r_str == str(node.left_child.right_child):
                return node.right_child

    # OR logic: Idempotenza & Assorbimento
    if isinstance(node, Or):
        l_str, r_str = str(node.left_child), str(node.right_child)
        if l_str == r_str: return node.left_child
        if isinstance(node.right_child, And):
            if l_str == str(node.right_child.left_child) or l_str == str(node.right_child.right_child):
                return node.left_child
        if isinstance(node.left_child, And):
            if r_str == str(node.left_child.left_child) or r_str == str(node.left_child.right_child):
                return node.right_child

    # TEMPORAL logic: Fusione Bound
    if isinstance(node, Globally) and isinstance(node.child, Globally):
        inner = node.child
        return Globally(
            node.left_time_bound + inner.left_time_bound,
            (0 if (node.right_unbound or inner.right_unbound) else node.right_time_bound + inner.right_time_bound),
            node.right_unbound or inner.right_unbound,
            inner.child
        )
    
    if isinstance(node, Eventually) and isinstance(node.child, Eventually):
        inner = node.child
        return Eventually(
            node.left_time_bound + inner.left_time_bound,
            (0 if (node.right_unbound or inner.right_unbound) else node.right_time_bound + inner.right_time_bound),
            node.right_unbound or inner.right_unbound,
            inner.child
        )

    return node

from train_arch import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula
from perturb import to_string
# Importa la funzione simplify che abbiamo appena definito

def test_simplify():
    test_cases = [
        # 1. Doppia Negazione
        ("not ( not ( x_0 >= 0.5 ) )", "x_0 >= 0.5000"),
        
        # 2. De Morgan
        ("not ( x_1 <= 0.2 or x_2 >= -1.0 )", "( not ( x_1 <= 0.2000 ) and not ( x_2 >= -1.0000 ) )"),
        
        # 3. Idempotenza e Assorbimento
        ("( x_0 <= 0.5 and x_0 <= 0.5 )", "x_0 <= 0.5000"),
        ("( x_0 <= 0.5 and ( x_0 <= 0.5 or x_1 >= 1.0 ) )", "x_0 <= 0.5000"),
        
        # 4. Annidamento Temporale (La prova del nove)
        ("always[10,20] ( always[5,5] ( x_0 >= 0.0 ) )", "always[15,25] ( x_0 >= 0.0000 )"),
        ("eventually[0,10] ( eventually[10,20] ( x_1 <= 1.0 ) )", "eventually[10,30] ( x_1 <= 1.0000 )"),
        
        # 5. Dualità e Negazione
        ("not ( eventually[10,20] ( not ( x_0 >= 0.5 ) ) )", "always[10,20] ( x_0 >= 0.5000 )"),
        
        # 6. Caso Complesso (Simile alle tue varianti ibride)
        ("not ( not ( ( x_1 >= 0.5 and x_1 >= 0.5 ) until[1,4] not ( not ( x_2 <= 1.5 ) ) ) )", 
         "( x_1 >= 0.5000 until[1,4] x_2 <= 1.5000 )")
    ]

    print(f"{'CASO':<50} | {'RISULTATO':<30} | {'STATUS'}")
    print("-" * 100)

    for raw, expected in test_cases:
        node = from_string_to_formula(raw)
        simplified_node = simplify(node)
        res_str = to_string(simplified_node)
        
        status = "✅ OK" if res_str.strip() == expected.strip() else "❌ FAIL"
        print(f"{raw[:48]:<50} | {res_str[:28]:<30} | {status}")
        if status == "❌ FAIL":
            print(f"   > Expected: {expected}")

# Esegui il test
# if __name__ == "__main__":
#     test_simplify()