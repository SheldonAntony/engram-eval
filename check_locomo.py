import json, urllib.request, os, re
url = 'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json'
cache = os.path.join(os.path.expanduser('~'), '.config', 'preflight', 'locomo10.json')
if not os.path.exists(cache):
    print('Downloading...')
    urllib.request.urlretrieve(url, cache)
    print('Done.')
else:
    print('Cache exists:', cache)
with open(cache, 'r', encoding='utf-8') as f:
    raw = json.load(f)
samples = list(raw) if isinstance(raw, list) else list(raw.values())
s0 = samples[0]
conv = s0.get('conversation', {})
keys = list(conv.keys())[:10]
nums = sorted(int(k.split('_')[1]) for k in conv.keys() if re.match(r'^session_\d+$', k))
t0 = conv['session_1'][0]
qa0 = s0['qa'][0]
print('Num samples:', len(samples))
print('Conv keys sample:', keys)
print('Session nums:', nums[:5])
print('First turn:', t0)
print('QA 0 keys:', list(qa0.keys()))
print('QA 0 category:', qa0.get('category'), '  type:', type(qa0.get('category')))
print('QA 0 answer:', str(qa0.get('answer'))[:80])
print('Total QA pairs:', sum(len(s.get('qa',[])) for s in samples))
