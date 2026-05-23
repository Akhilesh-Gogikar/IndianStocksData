# Data And Compliance Notes

These notes apply to every business model in this folder.

## Data Rights

- Treat scraped third-party pages as a validation source, not a commercial data license.
- Before selling APIs, dashboards, widgets, or reports, replace scraped-only dependencies with licensed data, public filings, exchange-permitted feeds, company disclosures, or explicit source permission.
- Track source URL, scrape time, parse version, and field-level freshness for every stored record.
- Do not expose raw third-party payloads to customers unless licensed.

## SEBI Boundary

Lower-risk wording:

- "Shows valuation metrics."
- "Flags unusual movement."
- "Compares peer financials."
- "Summarizes public disclosures."
- "Lets users define and test their own filters."

Higher-risk wording:

- "Buy this stock."
- "Sell now."
- "Target price."
- "Guaranteed return."
- "Best stock for your portfolio."
- "Personalized allocation."

Paid investment recommendations, research reports, model portfolios, or personalized portfolio advice should be treated as SEBI Research Analyst or Investment Adviser territory.

## Product Design Rule

Default to tools that help users inspect data and form their own view. Do not default to recommendations unless the company intentionally becomes a compliant regulated entity.
