# Deployment process

## Approvals

Production deploys require approval from the service owner. For changes
touching payment flows, a second approval from the platform lead is required.
Approvals are recorded on the merge request.

## Deploy windows

Deploys are allowed Monday to Thursday, 10:00–17:00. Friday and weekend
deploys are forbidden except for P1 incident fixes with an explicit
sign-off from the engineering manager. So no, you cannot deploy on Friday
evening unless production is on fire.

## Rollback

Every deploy must have a tested rollback path: previous image tag plus
reversible migrations. Irreversible migrations go through the two-step
expand/contract pattern.
