-- Seed Compliance Audit Logs

INSERT INTO compliance_logs (id, upload_id, user_id, action, details, ip_address, compliance_standard, created_at) VALUES
('log-1', 'upload-1', 'user-1', 'File Uploaded', 'Uploaded surveillance footage for processing', '192.168.1.100', 'GDPR', DATE_SUB(NOW(), INTERVAL 5 MINUTE)),
('log-2', 'upload-1', 'user-1', 'Processing Started', 'Started ML detection pipeline', '192.168.1.100', 'GDPR', DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('log-3', 'upload-1', 'user-1', 'Processing Completed', '12 detections found and anonymized', '192.168.1.100', 'GDPR', DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('log-4', 'upload-2', 'user-2', 'File Uploaded', 'Uploaded medical document for anonymization', '192.168.1.101', 'HIPAA', DATE_SUB(NOW(), INTERVAL 15 MINUTE)),
('log-5', 'upload-2', 'user-2', 'Processing Completed', '8 sensitive items detected and redacted', '192.168.1.101', 'HIPAA', DATE_SUB(NOW(), INTERVAL 14 MINUTE)),
('log-6', 'upload-4', 'user-1', 'File Uploaded', 'Conference photo uploaded for review', '192.168.1.100', 'CCPA', DATE_SUB(NOW(), INTERVAL 45 MINUTE)),
('log-7', 'upload-4', 'user-1', 'Manual Review', 'User reviewed and approved all 24 detections', '192.168.1.100', 'CCPA', DATE_SUB(NOW(), INTERVAL 43 MINUTE)),
('log-8', NULL, 'user-2', 'Report Generated', 'Monthly compliance report generated', '192.168.1.101', 'GDPR', DATE_SUB(NOW(), INTERVAL 1 DAY));
