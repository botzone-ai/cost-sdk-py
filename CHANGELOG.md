# Changelog

## 0.1.1 - 2026-06-29

### Fixed
- Events were rejected by the ingestion API (HTTP 400) and never reached the dashboard:
  - `occurredAt` now uses an RFC3339 `Z` suffix instead of a `+00:00` offset.
  - `userIdHash` / `route` / `featureTag` are omitted when unset rather than sent as `null`.
- The background queue no longer swallows server rejections. Non-2xx responses are
  retried when transient (408/425/429/5xx) and otherwise surfaced via the
  `botzone_cost` logger, with a new `IngestionQueue.failed_count()`.

## 0.1.0 - 2026-05-27

- Initial public release.
- Wraps Anthropic, OpenAI, and Google Generative AI SDKs.
- Sends metadata-only events to the Cost ingestion endpoint.
- Body capture (`capture_bodies`) is reserved for future parity with the TypeScript SDK and has no effect today.
