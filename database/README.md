# Database Schema & Migrations

This directory contains the PostgreSQL SQL migration scripts for Supabase.

## Migration Files

1. **[`misclassification_reports_table.sql`](misclassification_reports_table.sql)**:
   - Creates the `public.misclassification_reports` table for user-reported False Positives and False Negatives.
   - Sets up indexes on `report_type`, `status`, and `created_at`.
   - Enables Row Level Security (RLS) with policies for API insertion and querying.

2. **[`ntc_report_table.sql`](ntc_report_table.sql)**:
   - Creates the `public.ntc_report` table for regulatory smishing incident reporting (National Telecommunications Commission compliance).
   - Sets up indexes on `reported_at`, `verdict`, and `status`.
   - Enables Row Level Security (RLS) policies.

## How to Apply

1. Log into your [Supabase Dashboard](https://supabase.com).
2. Go to **SQL Editor**.
3. Copy and paste the contents of each `.sql` file and click **Run**.
