# Security policy

## Secret rotation

Service account secrets must be rotated every 60 days. Rotation evidence is
recorded in the `SEC-ROTATION` register, and expired credentials are revoked
within 4 hours.

## Access reviews

Privileged access is reviewed every 30 days by the security owner. Each review
must list the account, business justification, owner, and expiry date.

## Emergency access

A break-glass session may remain active for at most 2 hours. The incident
commander must link the session to a `SEC-ACCESS` ticket before activation.
