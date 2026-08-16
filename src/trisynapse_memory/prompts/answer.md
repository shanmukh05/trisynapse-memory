Answer using only the supplied trace records. Return strict JSON with fields answer and abstain.
Episode Recall summaries are never provided here.
Keep the answer concise and do not invent unsupported details.
For a factual question, give a short fact grounded in Trace evidence.
For a temporal question, resolve dates from observed_at and temporal anchors and return an absolute date when supported.
For a list question, return every distinct supported item rather than only the first match.
For an inference question, make the best supported judgment from all related Trace records and surface tensions explicitly.
