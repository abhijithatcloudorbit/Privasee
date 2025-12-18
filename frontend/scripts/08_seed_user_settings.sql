-- Seed User Settings

INSERT INTO user_settings (user_id, notifications_enabled, email_alerts, default_compliance_mode, language, timezone, theme) VALUES
('user-1', TRUE, TRUE, 'strict', 'en', 'Asia/Kolkata', 'dark'),
('user-2', TRUE, FALSE, 'moderate', 'en', 'Asia/Kolkata', 'dark'),
('user-3', FALSE, FALSE, 'custom', 'en', 'Asia/Kolkata', 'dark'),
('user-4', TRUE, TRUE, 'strict', 'en', 'Asia/Kolkata', 'light'),
('user-5', TRUE, TRUE, 'moderate', 'en', 'Asia/Kolkata', 'dark'),
('demo', TRUE, TRUE, 'strict', 'en', 'Asia/Kolkata', 'dark');
