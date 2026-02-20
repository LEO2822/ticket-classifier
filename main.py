import json

from src.ticket_classifier.inference import classify

ticket = "I want to cancel my credit card"

result = classify(ticket)
print(json.dumps(result, indent=2))
