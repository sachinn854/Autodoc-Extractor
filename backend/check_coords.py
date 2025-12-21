import json

with open('tmp/results/8a7e40f6-1e52-4b19-a38f-6718cefe9e79/ocr.json', 'r') as f:
    data = json.load(f)

tokens = data['pages'][0]['tokens']

print("Key tokens and their Y coordinates:")
for i, token in enumerate(tokens):
    text = token.get('text', '')
    bbox = token.get('bbox', [])
    
    if any(word in text for word in ['Tandoori', 'Lasooni', 'BIRYANI', '295.00', '275.00', '309.75', '288.75', 'Dal', 'Tadka', 'chicken']):
        print(f'{i}: "{text}" at Y={bbox[1]}, X={bbox[0]}')