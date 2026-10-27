## Classifier evaluation (20 labelled emails)

| Model | Intent acc. | Macro F1 | Priority acc. | To review | Wrong, not reviewed | Failures | Avg ms | p95 ms | Cost/email |
|---|---|---|---|---|---|---|---|---|---|
| fake | 55% | 0.58 | 75% | 45% | 3 | 0 | 0 | 1 | $0.00000 |
| openai:gpt-6-luna | 100% | 1.00 | 100% | 0% | 0 | 0 | 2937 | 6329 | $0.00009 |
| anthropic:claude-haiku-5-5 | 100% | 1.00 | 70% | 0% | 0 | 0 | 1715 | 2848 | $0.00014 |

### fake

| Intent | Precision | Recall | F1 | Emails |
|---|---|---|---|---|
| reply_needed | 50% | 60% | 0.55 | 5 |
| meeting_request | 50% | 50% | 0.50 | 4 |
| follow_up | 100% | 67% | 0.80 | 3 |
| fyi | 33% | 67% | 0.44 | 3 |
| newsletter | 100% | 50% | 0.67 | 2 |
| spam | 100% | 33% | 0.50 | 3 |

Confusion matrix (rows: expected, columns: predicted; 'failed' = no valid answer)

| expected \ predicted | reply | meeting | follow | fyi | newsletter | spam | failed |
|---|---|---|---|---|---|---|---|
| reply_needed | 3 | 1 | 0 | 1 | 0 | 0 | 0 |
| meeting_request | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| follow_up | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| fyi | 1 | 0 | 0 | 2 | 0 | 0 | 0 |
| newsletter | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| spam | 0 | 1 | 0 | 1 | 0 | 1 | 0 |

| Confidence | Emails | Intent accuracy |
|---|---|---|
| < 0.6 (review) | 9 | 33% |
| 0.6 to 0.8 | 9 | 67% |
| >= 0.8 | 2 | 100% |

Misclassified: v01 (reply_needed -> meeting_request, conf 0.50), v04 (reply_needed -> fyi, conf 0.40), v06 (meeting_request -> reply_needed, conf 0.65), v09 (meeting_request -> reply_needed, conf 0.65), v12 (follow_up -> fyi, conf 0.40), v14 (fyi -> reply_needed, conf 0.65), v17 (newsletter -> fyi, conf 0.40), v19 (spam -> fyi, conf 0.40), v20 (spam -> meeting_request, conf 0.50)

### openai:gpt-6-luna

| Intent | Precision | Recall | F1 | Emails |
|---|---|---|---|---|
| reply_needed | 100% | 100% | 1.00 | 5 |
| meeting_request | 100% | 100% | 1.00 | 4 |
| follow_up | 100% | 100% | 1.00 | 3 |
| fyi | 100% | 100% | 1.00 | 3 |
| newsletter | 100% | 100% | 1.00 | 2 |
| spam | 100% | 100% | 1.00 | 3 |

Confusion matrix (rows: expected, columns: predicted; 'failed' = no valid answer)

| expected \ predicted | reply | meeting | follow | fyi | newsletter | spam | failed |
|---|---|---|---|---|---|---|---|
| reply_needed | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| meeting_request | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| follow_up | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| fyi | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| newsletter | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| spam | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

| Confidence | Emails | Intent accuracy |
|---|---|---|
| < 0.6 (review) | 0 | - |
| 0.6 to 0.8 | 0 | - |
| >= 0.8 | 20 | 100% |

### anthropic:claude-haiku-5-5

| Intent | Precision | Recall | F1 | Emails |
|---|---|---|---|---|
| reply_needed | 100% | 100% | 1.00 | 5 |
| meeting_request | 100% | 100% | 1.00 | 4 |
| follow_up | 100% | 100% | 1.00 | 3 |
| fyi | 100% | 100% | 1.00 | 3 |
| newsletter | 100% | 100% | 1.00 | 2 |
| spam | 100% | 100% | 1.00 | 3 |

Confusion matrix (rows: expected, columns: predicted; 'failed' = no valid answer)

| expected \ predicted | reply | meeting | follow | fyi | newsletter | spam | failed |
|---|---|---|---|---|---|---|---|
| reply_needed | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| meeting_request | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| follow_up | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| fyi | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| newsletter | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| spam | 0 | 0 | 0 | 0 | 0 | 3 | 0 |

| Confidence | Emails | Intent accuracy |
|---|---|---|
| < 0.6 (review) | 0 | - |
| 0.6 to 0.8 | 0 | - |
| >= 0.8 | 20 | 100% |
