
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency

def calculate_association_rules(df, min_support=0.01):
    """
    Calculates Support, Confidence, Lift, and Chi-Square for Sub-Category pairs.
    """
    
    baskets = df.groupby('Order ID')['Sub-Category'].apply(lambda x: list(set(x)))
    n_transactions = len(baskets)
    
    
    item_counts = df.groupby('Sub-Category')['Order ID'].nunique()
    item_support = item_counts / n_transactions
    
    
    pair_counts = {}
    for basket in baskets:
        if len(basket) > 1:
            for pair in combinations(sorted(basket), 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    
    results = []
    for pair, frequency in pair_counts.items():
        item_a, item_b = pair
        support_ab = frequency / n_transactions
        
        
        if support_ab < min_support:
            continue
            
        support_a = item_support[item_a]
        support_b = item_support[item_b]
        
        lift = support_ab / (support_a * support_b)
        confidence_a_to_b = support_ab / support_a
        confidence_b_to_a = support_ab / support_b
        

        count_ab = frequency
        count_a_not_b = item_counts[item_a] - count_ab
        count_b_not_a = item_counts[item_b] - count_ab
        count_neither = n_transactions - item_counts[item_a] - item_counts[item_b] + count_ab
        
        obs = np.array([[count_ab, count_a_not_b], [count_b_not_a, count_neither]])
        chi2, p, _, _ = chi2_contingency(obs)
        
        results.append({
            'Item A': item_a,
            'Item B': item_b,
            'Support': support_ab,
            'Lift': lift,
            'Confidence (A->B)': confidence_a_to_b,
            'Chi-Square': chi2,
            'P-Value': p
        })
        
    rules_df = pd.DataFrame(results)
    return rules_df.sort_values(by='Lift', ascending=False)