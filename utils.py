import copy
import random
from typing import List
from train import Atom, Not, And, Or, Globally, Eventually, Until, from_string_to_formula

MAX_DEPTH = 25 

def simplify(formula, depth = 0):

    """

    Recursively simplifies STL formulae using logical equivalences and temporal rules.

    """

    if depth > MAX_DEPTH:

        return formula

    try: node = copy.deepcopy(formula)

    except: return formula

    # --- Recursively simplify children first ---

    if hasattr(node, "child"):

        node.child = simplify(node.child, depth + 1)

    elif hasattr(node, "left_child") and hasattr(node, "right_child"):

        node.left_child = simplify(node.left_child, depth + 1)

        node.right_child = simplify(node.right_child, depth + 1)


    # --- NOT operator simplifications ---

    if isinstance(node, Not):

        child = node.child

        if isinstance(child, Not):

            return simplify(child.child, depth + 1)  # Double negation: ¬(¬φ) → φ

        elif isinstance(child, Atom):

            return Atom(  # Negated predicate

                var_index=child.var_index,

                threshold=copy.deepcopy(child.threshold),

                lte=not child.lte,

            )

        elif isinstance(child, Eventually):

            return simplify(

                Globally(  # ¬F_I(φ) → G_I(¬φ)

                    unbound=child.unbound,

                    right_unbound=child.right_unbound,

                    left_time_bound=child.left_time_bound,

                    right_time_bound=child.right_time_bound,

                    child=Not(child.child),

                ), depth + 1

            )

        elif isinstance(child, Globally):

            return simplify(

                Eventually(  # ¬G_I(φ) → F_I(¬φ)

                    unbound=child.unbound,

                    right_unbound=child.right_unbound,

                    left_time_bound=child.left_time_bound,

                    right_time_bound=child.right_time_bound,

                    child=Not(child.child),

                ), depth + 1

            )

        elif isinstance(child, Or):

            # ¬(a ∨ b) → ¬a ∧ ¬b

            return simplify(

                And(

                    left_child=simplify(Not(child.left_child), depth + 1),

                    right_child=simplify(Not(child.right_child), depth + 1),

                ), depth + 1

            )

        elif isinstance(child, And):

            # ¬(a ∧ b) → ¬a ∨ ¬b

            return simplify(

                Or(

                    left_child=simplify(Not(child.left_child), depth + 1),

                    right_child=simplify(Not(child.right_child), depth + 1),

                ), depth + 1

            )


    # --- AND simplifications ---

    if isinstance(node, And):

        l, r = node.left_child, node.right_child

        if l == r:

            return simplify(l, depth + 1)  # a ∧ a → a


        # a ∧ (a ∨ b) → a

        if isinstance(r, Or) and (l == r.left_child or l == r.right_child):

            return simplify(l, depth + 1)

        if isinstance(l, Or) and (r == l.left_child or r == l.right_child):

            return simplify(r, depth + 1)


        # a ∧ (a ∧ b) → a ∧ b

        if isinstance(r, And):

            if l == r.left_child:

                return simplify(And(left_child=l, right_child=r.right_child), depth + 1)

            if l == r.right_child:

                return simplify(And(left_child=l, right_child=r.left_child), depth + 1)

        if isinstance(l, And):

            if r == l.left_child:

                return simplify(And(left_child=r, right_child=l.right_child), depth + 1)

            if r == l.right_child:

                return simplify(And(left_child=r, right_child=l.left_child), depth + 1)


    # --- OR simplifications ---

    if isinstance(node, Or):

        l, r = node.left_child, node.right_child

        if l == r:

            return simplify(l, depth + 1)  # a ∨ a → a


        # a ∨ (a ∧ b) → a

        if isinstance(r, And) and (l == r.left_child or l == r.right_child):

            return simplify(l, depth + 1)

        if isinstance(l, And) and (r == l.left_child or r == l.right_child):

            return simplify(r, depth + 1)


        # a ∨ (a ∨ b) → a ∨ b

        if isinstance(r, Or):

            if l == r.left_child:

                return simplify(Or(left_child=l, right_child=r.right_child), depth + 1)

            if l == r.right_child:

                return simplify(Or(left_child=l, right_child=r.left_child), depth + 1)

        if isinstance(l, Or):

            if r == l.left_child:

                return simplify(Or(left_child=r, right_child=l.right_child), depth + 1)

            if r == l.right_child:

                return simplify(Or(left_child=r, right_child=l.left_child), depth + 1)


    # --- Nested Globally simplification: G_I(G_J(φ)) → G_{I+J}(φ) ---

    if isinstance(node, Globally) and isinstance(node.child, Globally):

        inner = node.child

        child = inner.child

        return simplify(

            Globally(

                unbound=node.unbound or inner.unbound,

                right_unbound=node.right_unbound or inner.right_unbound,

                left_time_bound=node.left_time_bound + inner.left_time_bound,

                right_time_bound=(

                    1

                    if (node.right_unbound or inner.right_unbound)

                    else node.right_time_bound + inner.right_time_bound - 1

                ),

                child=copy.deepcopy(child),

            ), depth + 1

        )


    # --- Nested Eventually simplification: F_I(F_J(φ)) → F_{I+J}(φ) ---

    if isinstance(node, Eventually) and isinstance(node.child, Eventually):

        inner = node.child

        child = inner.child

        return simplify(

            Eventually(

                unbound=node.unbound or inner.unbound,

                right_unbound=node.right_unbound or inner.right_unbound,

                left_time_bound=node.left_time_bound + inner.left_time_bound,

                right_time_bound=(

                    1

                    if (node.right_unbound or inner.right_unbound)

                    else node.right_time_bound + inner.right_time_bound - 1

                ),

                child=copy.deepcopy(child),

            ), depth + 1

        )


    # --- Until simplification: φ U φ → φ ---

    if isinstance(node, Until) and node.left_child == node.right_child:

        return simplify(node.left_child, depth + 1)


    return node
