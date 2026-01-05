from data import generate_variants

def run_tests():
    test_formulas = [
        # TEST 1: Annidamento temporale (Verifica drift bound)
        "always[0,10] ( always[5,5] ( always[0,2] ( x_0 <= 0.5 ) ) )",
        
        # TEST 2: Annidamento binario (Verifica crash parser deque)
#        "not ( ( x_0 <= 1.0 or x_1 >= 2.0 ) or ( x_2 <= 0.0 or x_1 >= 5.0 ) )",
        
        # TEST 3: Idempotenza e ridondanza (Verifica semplificazione)
#        "eventually[0,10] ( ( x_0 <= 0.5 and x_0 <= 0.5 ) or ( x_0 <= 0.5 ) )",
        
        # TEST 4: Negazioni e Dualità (Verifica equivalenza semantica not always not -> eventually)
        "not ( always[0,100] ( not ( eventually[0,10] ( x_1 >= 10.0 ) ) ) )"
    ]

    for i, original in enumerate(test_formulas, 1):
        print(f"\n🔥 TEST {i} - ORIGINALE: {original}")
        variants = generate_variants(original, num_variants=5)
        
        if not variants:
            print(f"   ⚠️ Nessuna variante generata (probabile crash silente nel parser originale)")
        else:
            for j, v in enumerate(variants, 1):
                print(f"  {j}. {v}")

if __name__ == "__main__":
    run_tests()
