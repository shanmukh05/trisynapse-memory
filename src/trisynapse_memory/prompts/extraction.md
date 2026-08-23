Extract candidate factual statements for an append-only memory trace.
Each input record has an immutable observation ID. Return strict JSON:
{"facts":[{"subject":str,"relation":str,"object":str,"text":str,"temporal_expression":str|null,"confidence":float,"evidence_ids":[str]}]}.
One fact per claim. Every evidence_ids value must be an observation ID supplied in the input and must contain direct support for that fact. Use the smallest sufficient evidence set. Resolve relative dates when possible. Do not deduplicate, merge, replace, or adjudicate conflicts.
