import json

with open('tmp/results/136c52de-9941-41ac-8f94-bb30c6c11e06/ocr.json', 'r') as f:
    data = json.load(f)

tokens = data['pages'][0]['tokens']

print("Looking for Lasooni Dal Tadka tokens and number 14:")
for i, token in enumerate(tokens):
    text = token.get('text', '')
    bbox = token.get('bbox', [])
    
    if 'Lasooni' in text or 'Dal' in text or 'Tadka' in text or text == '14' or text == '1':
        print(f'Token {i}: "{text}" at bbox {bbox}')