# Data Quality Policy

Production risk pipelines must validate schema completeness, primary-key uniqueness, foreign-key integrity, null percentage, freshness, and outlier thresholds.

Critical tables include customers, loan applications, disbursements, repayments, bureau features, and collection calls.

A data-quality incident must be raised when null rate exceeds 5 percent in critical fields, duplicate primary keys are found, foreign-key failure rate exceeds 1 percent, or month-on-month metric movement exceeds expected thresholds.

Every incident summary should include impacted table, failed rule, severity, suspected root cause, business impact, and recommended owner action.
