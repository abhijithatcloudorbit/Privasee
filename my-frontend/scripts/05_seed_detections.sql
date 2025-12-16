-- Seed Sample Detections

INSERT INTO detections (id, upload_id, model_id, detection_type, confidence_score, bounding_box_x, bounding_box_y, bounding_box_width, bounding_box_height, status, blur_applied, detected_at) VALUES
-- Upload 1 detections
('det-1', 'upload-1', 'model-face', 'face', 98.00, 15.00, 20.00, 12.00, 15.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('det-2', 'upload-1', 'model-face', 'face', 96.00, 45.00, 25.00, 10.00, 13.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('det-3', 'upload-1', 'model-person', 'person', 95.00, 60.00, 30.00, 20.00, 35.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('det-4', 'upload-1', 'model-license', 'license-plate', 99.00, 25.00, 65.00, 8.00, 4.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 4 MINUTE)),
('det-5', 'upload-1', 'model-text', 'text', 94.00, 70.00, 10.00, 15.00, 5.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 4 MINUTE)),

-- Upload 2 detections
('det-6', 'upload-2', 'model-medical', 'medical', 97.50, 10.00, 15.00, 30.00, 40.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 14 MINUTE)),
('det-7', 'upload-2', 'model-text', 'text', 95.00, 20.00, 60.00, 25.00, 8.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 14 MINUTE)),
('det-8', 'upload-2', 'model-face', 'face', 93.00, 50.00, 10.00, 8.00, 10.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 14 MINUTE)),

-- Upload 4 detections
('det-9', 'upload-4', 'model-face', 'face', 98.50, 20.00, 25.00, 10.00, 12.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 43 MINUTE)),
('det-10', 'upload-4', 'model-face', 'face', 97.00, 35.00, 28.00, 9.00, 11.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 43 MINUTE)),
('det-11', 'upload-4', 'model-person', 'person', 96.50, 15.00, 30.00, 25.00, 40.00, 'approved', TRUE, DATE_SUB(NOW(), INTERVAL 43 MINUTE));
