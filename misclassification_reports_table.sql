-- SQL Migration: Create public.misclassification_reports table and setup Row Level Security (RLS) policies

CREATE TABLE IF NOT EXISTS public.misclassification_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender VARCHAR(50) DEFAULT 'UNKNOWN',
    message TEXT NOT NULL,
    has_url BOOLEAN DEFAULT false,
    extracted_url TEXT,
    original_verdict VARCHAR(50) NOT NULL,
    original_score DOUBLE PRECISION NOT NULL,
    original_ml_score DOUBLE PRECISION,
    original_dl_score DOUBLE PRECISION,
    user_verdict VARCHAR(50) NOT NULL,
    report_type VARCHAR(50) NOT NULL, -- 'false_positive' or 'false_negative'
    user_comment TEXT,
    app_version VARCHAR(50),
    device_id VARCHAR(100),
    status VARCHAR(50) DEFAULT 'pending_review', -- 'pending_review', 'verified', 'dismissed'
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for efficient querying, filtering, and retraining workflows
CREATE INDEX IF NOT EXISTS idx_misclass_report_type ON public.misclassification_reports (report_type);
CREATE INDEX IF NOT EXISTS idx_misclass_status ON public.misclassification_reports (status);
CREATE INDEX IF NOT EXISTS idx_misclass_created_at ON public.misclassification_reports (created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE public.misclassification_reports ENABLE ROW LEVEL SECURITY;

-- Grant INSERT permission to anon and authenticated API clients
DROP POLICY IF EXISTS "Allow insert to misclassification_reports" ON public.misclassification_reports;
CREATE POLICY "Allow insert to misclassification_reports"
ON public.misclassification_reports
FOR INSERT
TO anon, authenticated
WITH CHECK (true);

-- Grant SELECT permission to anon and authenticated API clients
DROP POLICY IF EXISTS "Allow select to misclassification_reports" ON public.misclassification_reports;
CREATE POLICY "Allow select to misclassification_reports"
ON public.misclassification_reports
FOR SELECT
TO anon, authenticated
USING (true);
