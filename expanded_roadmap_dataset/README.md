# Expanded Synthetic Text-to-Roadmap Dataset

## Contents
- `train.jsonl` — 8,000 records across 800 maps
- `validation.jsonl` — 1,000 records across 100 maps
- `test.jsonl` — 1,000 records across 100 maps
- `maps.jsonl` — full synthetic map inventory with grid coordinates, spatial relations, walkable connections, and descriptions
- `schema.json` — schema, semantics, and caveats

## Record design
Each line is a JSON object with:
- `id`, `map_id`
- `input.map_description`
- `input.locations`
- `input.spatial_relations`
- `input.walkable_connections`
- `input.query` (start and destination)
- optional `input.road_closures`
- `target.case_type` (`route`, `ambiguous`, `unreachable`)
- `target.route`, `target.route_text`, `target.reachable`, `target.ambiguity_reason`

## Recommended use for LLM fine-tuning
Use the input object as the user message and the target object as the assistant response. For chat fine-tuning, transform each record into:
{"messages":[
  {"role":"system","content":"You are a navigation assistant. Use only explicit walkable connections; distinguish unknown information from a proven unreachable destination. Return a valid JSON answer."},
  {"role":"user","content":"<serialized input JSON>"},
  {"role":"assistant","content":"<serialized target JSON>"}
]}
This is a template; a preformatted chat JSONL is not included so you can adapt system instructions to your model.

## Data split
Split by whole map ID to test generalization to unseen maps:
- train: 800 maps
- validation: 100 maps
- test: 100 maps

## Caveats
This is synthetic training data and may contain simplified map assumptions. Run automated graph validation and manual spot checks before production use. In production, use a pathfinding algorithm to verify model-generated routes.
