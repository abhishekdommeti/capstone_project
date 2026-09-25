# Example /ask calls (MOCK_LLM left at its default)

### Request
```json
{
  "query": "What is the delivery fee if my order is under INR 149?"
}
```
### Response
```json
{
  "answer": "Based on the retrieved context: \ufeffZepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard de",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_07"
  ],
  "confidence": 1.0
}
```

### Request
```json
{
  "query": "What's your favorite color?"
}
```
### Response
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```
