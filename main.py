import json

from ticket_classifier import classify

ticket = "I forgot my password and I can't log in to my account. Can you help me reset it?"

result = classify(ticket)
print(json.dumps(result, indent=2))
