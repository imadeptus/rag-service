# On-call and incident policy

## Incident response SLA

Acknowledgement SLA: 15 minutes for P1, 1 hour for P2, next business day for
P3. Resolution targets: P1 — 4 hours, P2 — 1 business day. The on-call
engineer owns the incident until an explicit handover is recorded in the
incident channel.

## Audit logs

Audit logs are retained for 400 days in cold storage. Access requires a
ticket approved by the security team. Logs older than 400 days are deleted
automatically by the retention job.

## Escalation

If the on-call engineer does not acknowledge within the SLA, PagerDuty
escalates to the secondary, then to the engineering manager.
