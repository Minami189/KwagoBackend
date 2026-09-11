-- SQL Migration: Create public.ntc_report table and setup Row Level Security (RLS) policies

CREATE TABLE IF NOT EXISTS public.ntc_report (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender VARCHAR(50) DEFAULT 'UNKNOWN',
    message TEXT NOT NULL,
    url_analysis JSONB,
    ml_score DOUBLE PRECISION DEFAULT 0.0,
    dl_score DOUBLE PRECISION DEFAULT 0.0,
    final_score DOUBLE PRECISION DEFAULT 0.0,
    verdict VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    reported_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create indexes for efficient querying and reporting
CREATE INDEX IF NOT EXISTS idx_ntc_report_reported_at ON public.ntc_report (reported_at DESC);
CREATE INDEX IF NOT EXISTS idx_ntc_report_verdict ON public.ntc_report (verdict);
CREATE INDEX IF NOT EXISTS idx_ntc_report_status ON public.ntc_report (status);

-- Enable Row Level Security (RLS)
ALTER TABLE public.ntc_report ENABLE ROW LEVEL SECURITY;

-- Grant INSERT permission to anon and authenticated API clients
DROP POLICY IF EXISTS "Allow insert to ntc_report" ON public.ntc_report;
CREATE POLICY "Allow insert to ntc_report"
ON public.ntc_report
FOR INSERT
TO anon, authenticated
WITH CHECK (true);

-- Grant SELECT permission to anon and authenticated API clients
DROP POLICY IF EXISTS "Allow select to ntc_report" ON public.ntc_report;
CREATE POLICY "Allow select to ntc_report"
ON public.ntc_report
FOR SELECT
TO anon, authenticated
USING (true);
