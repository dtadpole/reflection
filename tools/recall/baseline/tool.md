# Recall

## Description
Look up the raw content of a problem, experience, or card by its ID.
Returns the full JSON representation stored on the filesystem.

## Input Schema
- entity_type (str, required): Type of entity to recall — "problem", "experience", or "card"
- entity_id (str, required): The unique identifier (ULID) of the entity

## Output Schema
- entity_type (str): The type that was looked up
- entity_id (str): The ID that was looked up
- data (object): The full JSON content of the entity

## Examples
Input:
```json
{
  "entity_type": "problem",
  "entity_id": "01JQXYZ..."
}
```

Output:
```json
{
  "entity_type": "problem",
  "entity_id": "01JQXYZ...",
  "data": {
    "problem_id": "01JQXYZ...",
    "title": "Fused Softmax Kernel",
    "description": "Write a Triton kernel...",
    "domain": "triton_kernels",
    "difficulty": "medium",
    "test_cases": [...]
  }
}
```
