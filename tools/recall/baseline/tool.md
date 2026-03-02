# Recall

## Description
Three tools for accessing raw data from the filesystem:
- **fetch** — Look up the full content of a problem, experience, or card by its ID
- **outline** — Get structure info: format (json/jsonl), and for jsonl, message count and lengths
- **excerpt** — Read specific rows from a JSONL experience file by row range

## fetch — Entity Lookup

### Input Schema
- entity_type (str, required): Type of entity — "problem", "experience", or "card"
- entity_id (str, required): The unique identifier (ULID) of the entity

### Output Schema
- entity_type (str): The type that was looked up
- entity_id (str): The ID that was looked up
- format (str): "json" or "jsonl"
- data (object): The full content (JSON object or raw JSONL string for experiences)

## outline — Entity Structure

### Input Schema
- entity_type (str, required): Type of entity — "problem", "experience", or "card"
- entity_id (str, required): The unique identifier (ULID) of the entity

### Output Schema (jsonl — experiences)
- entity_type, entity_id, format: "jsonl"
- total_messages (int): Number of messages
- total_length (int): Sum of all message lengths
- messages (array): Each with row, role, length

### Output Schema (json — problems, cards)
- entity_type, entity_id, format: "json"
- length (int): Character length of the JSON content

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
