# Payments integration runbook

## API token rotation

To rotate the API token for the payments integration: generate a new token in
the partner portal, update the `PAYMENTS_API_TOKEN` secret in the vault, and
restart the payments worker. The old token stays valid for 24 hours, so
rotation is zero-downtime. Rotate at least every 90 days.

## Staging environment

The staging database listens on port 5433 (not the default 5432, to avoid
accidental writes from local tools). Connection details live in the vault
under `staging/payments-db`.

## Webhook failures

When a webhook delivery fails, the provider retries with exponential backoff:
1 minute, 5 minutes, 30 minutes, 2 hours, then gives up. Failed deliveries
after all retries land in the dead letter queue and trigger a `webhook-dlq`
alert. Replay them with `scripts/replay_webhooks.py --since <timestamp>`.
