# Weighted Final Verdict Plan for VirusTotal URL Scans

## Goal
Create a single final verdict from many VirusTotal engine results using a weighted score.

## Data inputs
- `last_analysis_results` from VirusTotal scan response
- Each engine/source provides a `category` and `result`
- Relevant categories: `malicious`, `suspicious`, `undetected`, `harmless`, `type-unsupported`, `timeout`

## Scoring model
1. Normalize each engine verdict to a numeric score:
   - `malicious` = 1.0
   - `suspicious` = 0.75
   - `undetected` = 0.0
   - `harmless` = 0.0
   - `type-unsupported` = 0.1
   - `timeout` = 0.05

2. Apply a weight for each source based on its reputation, historical accuracy, and reliability.

3. Compute the weighted score:
   - `weighted_score = sum(score_i * weight_i) / sum(weight_i)`

4. Determine the final verdict using thresholds:
   - `weighted_score >= 0.65` → `malicious`
   - `0.35 <= weighted_score < 0.65` → `suspicious`
   - `weighted_score < 0.35` → `benign`

## Additional adjustments
- Use a separate `detection_count` multiplier if many engines say `malicious`.
- Boost the score when high-weight vendors report `malicious`.
- Reduce the score for a preponderance of `undetected` or `harmless` responses.

## Implementation notes
- Extract `name`, `category`, and `result` from each engine entry.
- Convert textual verdicts to scores before weighting.
- Keep the weight table external so it can be tuned without changing logic.

## Example workflow
1. Load the VirusTotal JSON response.
2. Retrieve `response['data']['attributes']['last_analysis_results']`.
3. For each engine:
   - Map `category` to numeric score.
   - Multiply by engine weight.
4. Sum weighted scores and normalize.
5. Apply thresholds to return a final verdict.

## Why this works
- It treats each engine's judgment proportionally.
- It gives more influence to trusted, accurate engines.
- It avoids a simple majority vote, which can be skewed by many low-confidence sources.
- It is adaptable: weights and thresholds can be tuned over time.
