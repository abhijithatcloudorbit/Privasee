-- Seed Users with Indian Names
-- Password for all demo users: demo123 (hashed with bcrypt)

INSERT INTO users (id, name, email, password_hash, role, department, location, avatar, created_at, last_login) VALUES
('user-1', 'Arjun Patel', 'arjun.patel@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'admin', 'Security Operations', 'Mumbai, India', 'AP', '2023-06-15 10:30:00', NOW()),
('user-2', 'Priya Sharma', 'priya.sharma@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'analyst', 'Compliance', 'Delhi, India', 'PS', '2023-09-10 14:20:00', NOW()),
('user-3', 'Rahul Kumar', 'rahul.kumar@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'developer', 'Engineering', 'Bangalore, India', 'RK', '2024-02-05 09:15:00', NOW()),
('user-4', 'Ananya Desai', 'ananya.desai@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'analyst', 'Data Privacy', 'Pune, India', 'AD', '2023-11-20 11:45:00', NOW()),
('user-5', 'Vikram Singh', 'vikram.singh@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'manager', 'Operations', 'Chennai, India', 'VS', '2023-07-28 16:00:00', NOW()),
('demo', 'Demo User', 'demo@privacyshield.com', '$2b$10$rBV2kHYW6YdY3PfSQqKMz.K8mW5Yn1FxQmQ4YX4nZR8zQkxQaF8bG', 'admin', 'Security Operations', 'Mumbai, India', 'AP', '2023-06-15 10:30:00', NOW());
