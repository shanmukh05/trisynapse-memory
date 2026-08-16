Extract candidate factual statements for an append-only memory trace.
Return strict JSON: {"facts":[{"subject":str,"relation":str,"object":str,"text":str,"temporal_expression":str|null,"confidence":float}]}.
One fact per claim. Resolve relative dates when possible. Do not deduplicate, merge, replace, or adjudicate conflicts.
