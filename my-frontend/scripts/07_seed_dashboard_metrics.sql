-- Seed Dashboard Metrics

INSERT INTO dashboard_metrics (id, user_id, metric_type, metric_value, metadata, calculated_at) VALUES
('metric-1', NULL, 'total_uploads', 1247, '{"period": "all_time"}', NOW()),
('metric-2', NULL, 'total_detections', 18392, '{"period": "all_time"}', NOW()),
('metric-3', NULL, 'active_processing', 3, '{"period": "current"}', NOW()),
('metric-4', NULL, 'compliance_score', 98.70, '{"period": "last_30_days"}', NOW()),
('metric-5', NULL, 'avg_processing_time', 3.20, '{"period": "last_30_days", "unit": "seconds"}', NOW()),
('metric-6', NULL, 'faces_detected', 8234, '{"period": "all_time", "type": "face"}', NOW()),
('metric-7', NULL, 'people_detected', 5621, '{"period": "all_time", "type": "person"}', NOW()),
('metric-8', NULL, 'text_detected', 2845, '{"period": "all_time", "type": "text"}', NOW()),
('metric-9', NULL, 'medical_detected', 1203, '{"period": "all_time", "type": "medical"}', NOW()),
('metric-10', NULL, 'license_plates_detected', 489, '{"period": "all_time", "type": "license-plate"}', NOW());
