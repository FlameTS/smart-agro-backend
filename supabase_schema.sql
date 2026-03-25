-- ============================================
-- Smart Agro Database Schema for Supabase
-- Copy and paste all of this into Supabase SQL Editor
-- ============================================

-- Create crops table
CREATE TABLE crops (
  id BIGSERIAL PRIMARY KEY,
  crop_name TEXT NOT NULL UNIQUE,
  info TEXT DEFAULT 'Crop information will be added later.',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create diseases table
CREATE TABLE diseases (
  id BIGSERIAL PRIMARY KEY,
  crop_id BIGINT NOT NULL REFERENCES crops(id) ON DELETE CASCADE,
  disease_name TEXT NOT NULL,
  description TEXT DEFAULT 'Details will be added later.',
  cure TEXT DEFAULT 'Treatment information will be added later.',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(crop_id, disease_name)
);

-- Create prediction history table (for analytics)
CREATE TABLE prediction_history (
  id BIGSERIAL PRIMARY KEY,
  crop_name TEXT NOT NULL,
  disease_name TEXT NOT NULL,
  confidence DECIMAL(5,4) NOT NULL,
  image_uploaded BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_diseases_crop_id ON diseases(crop_id);
CREATE INDEX idx_prediction_history_crop ON prediction_history(crop_name);
CREATE INDEX idx_prediction_history_disease ON prediction_history(disease_name);
CREATE INDEX idx_prediction_history_created ON prediction_history(created_at);

-- Enable Row Level Security (RLS)
ALTER TABLE crops ENABLE ROW LEVEL SECURITY;
ALTER TABLE diseases ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_history ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (allow public read, authenticated write)
CREATE POLICY "allow_anon_read" ON crops
  FOR SELECT USING (true);

CREATE POLICY "allow_anon_read" ON diseases
  FOR SELECT USING (true);

CREATE POLICY "allow_anon_read" ON prediction_history
  FOR SELECT USING (true);

CREATE POLICY "allow_insert_predictions" ON prediction_history
  FOR INSERT WITH CHECK (true);
