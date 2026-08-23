Answer using only the supplied trace records. Return strict JSON with fields answer, abstain, and citation_ids.
Episode Recall summaries are never provided here.
Keep the answer concise and do not invent unsupported details.
Abstain only when the supplied records contain no relevant evidence. Do not abstain merely because answering requires connecting multiple supplied records.
`citation_ids` must contain only the IDs of the smallest sufficient set of supplied trace records that directly supports the answer. Return an empty list when abstaining.
For a factual question, give a short fact grounded in Trace evidence.
For a temporal question, resolve dates from observed_at and temporal anchors and return an absolute date when supported.
For a list question, return every distinct supported item rather than only the first match.
For an inference question, make the best supported judgment from all related Trace records and surface tensions explicitly.
