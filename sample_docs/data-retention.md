# Data retention policy

## Backups

Daily database backups are retained for 35 days. Monthly backups are retained
for 13 months. Restore verification runs once per quarter against an isolated
environment.

## Deletion

Expired backups are deleted within 24 hours after the retention period ends.
Deletion reports are stored in the `RETENTION-AUDIT` archive.

## Legal holds

A legal hold pauses scheduled deletion only when a compliance officer records
the exception in a `LEGAL-HOLD` ticket. The ticket must name the dataset and
the hold expiry date.
