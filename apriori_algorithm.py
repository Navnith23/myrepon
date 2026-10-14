
from itertools import combinations

transactions = [
    {'I1', 'I2', 'I5'},
    {'I2', 'I4'},
    {'I2', 'I3'},
    {'I1', 'I2', 'I4'},
    {'I1', 'I3'},
    {'I2', 'I3'},
    {'I1', 'I3'},
    {'I2', 'I3'},
    {'I1', 'I3'},
    {'I1', 'I2', 'I3', 'I5'},
    {'I1', 'I2', 'I3'}
]
min_support = 2
min_confidence = 0.6

def support(itemset):
    return sum(itemset.issubset(t) for t in transactions)

items = set(item for t in transactions for item in t)
frequent={}

for item in items:
    count = support({item})
    if count >= min_support:
        frequent[frozenset([item])] = count

k =2 
while True:
    previous = list(frequent.keys())
    candidates = set()
    for a,b in combinations(previous,2):
        union = a | b
        if len(union) == k:
            if all(frozenset(s) in frequent for s in combinations(union,k-1)):
                candidates.add(union)
    new_frequent = {}
    for itemset in candidates:
        count = support(itemset)
        if count >= min_support:
            new_frequent[itemset] = count
    if not new_frequent:
        break
    frequent.update(new_frequent)
    k += 1
print("Frequent Itemsets:")
for itemset,count in frequent.items():
    print("Itemset:",set(itemset),"Support:",count)

print("\nAssociation Rules:")
for itemset,count in frequent.items():
    if len(itemset) > 1:
        for left_side_size in range(1,len(itemset)):
            for left_items in combinations(itemset,left_side_size):
                left_set = frozenset(left_items)
                right_set = itemset - left_set
                confidence = count /frequent[left_set]
                if confidence >= min_confidence:
                    print(f"{set(left_set)} --> {set(right_set)} Confidence: {round(confidence,2)} Support: {count}")

            




