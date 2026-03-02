# Recall

## Description
Two tools for accessing raw data from the filesystem:
- **recall** — Look up the full content of a problem, experience, or card by its ID
- **excerpt** — Read specific rows from a JSONL experience file by row range

## recall — Entity Lookup

### Input Schema
- entity_type (str, required): Type of entity — "problem", "experience", or "card"
- entity_id (str, required): The unique identifier (ULID) of the entity

### Output Schema
- entity_type (str): The type that was looked up
- entity_id (str): The ID that was looked up
- data (object): The full JSON content (or raw JSONL string for experiences)

### Examples
```json
{"entity_type": "problem", "entity_id": "01JQXYZ..."}
```

## excerpt — JSONL Row Reader

### Input Schema
- experience_id (str, required): The experience ID
- start_row (int, optional): First row to read (1-based, default 1)
- end_row (int, optional): Last row to read (inclusive, default last row)

### Output Schema
- experience_id (str): The experience ID
- start_row (int): Actual start row returned
- end_row (int): Actual end row returned
- total_rows (int): Total rows in the file
- rows (array): Parsed JSON objects for each row

### Examples
```json
{"experience_id": "01JQXYZ...", "start_row": 7, "end_row": 15}
```
