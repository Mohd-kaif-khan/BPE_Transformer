import json

def get_stats(ids):
    count = {}

    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0)+1
    return count


def merge(ids, pair, idx):
    
    newids = []
    i = 0

    while(i < len(ids)):
        if(i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def encode(text, type):
    
    path = ''
    if type == 'encoder':
        path = 'English_Merges.json'
    else:
        path = 'Hindi_Merges.json'

    # print(path)
    with open(path, 'r') as f:
        data = json.load(f)

    merges = {eval(key):val for key , val in data.items()}

    tokens = list(text.encode('utf-8'))

    while(len(tokens) > 2):
        stats = get_stats(tokens)
        pair = min(stats, key=stats.get)
        # print(pair)
        if pair not in merges:
            break
        
        tokens = merge(tokens, pair, merges[pair])
    return tokens

def decode(ids, type):
    
    path = ''
    if type == 'encoder':
        path = 'index_to_english.json'
    else:
        path = 'index_to_hindi.json'

    with open(path, 'r') as f:
        data = json.load(f)

    vocab = {eval(key):eval(val) for key , val in data.items()}

    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode('utf-8', errors='replace')

    return text

