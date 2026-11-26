# Nemotron embedding input admission

The native `text_to_embedding` task accepts one nonempty UTF-8 input with a
default, query, or document role. It adds the checkpoint's role prefix, tokenizes
the resulting text, and rejects token counts outside the built sequence profile.
The engine's sequence length limits inference tokens, not the original text size
or the cost of Unicode normalization and tokenization.

Applications exposing this SDK task to untrusted clients must enforce a raw
UTF-8 byte limit and a preprocessing-work budget before copying requests or
invoking the SDK. Choose these admission limits for the application's memory,
latency, and concurrency requirements. The native task has no configurable
request-byte budget and does not provide request cancellation during tokenization.
The public SDK serializes execution on a model handle, so expensive preprocessing
can delay other requests sharing that handle. Host admission control is required
even when a request will eventually exceed the token profile.
