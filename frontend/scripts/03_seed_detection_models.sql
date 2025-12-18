-- Seed Detection Models

INSERT INTO detection_models (id, name, type, enabled, accuracy_score, description) VALUES
('model-face', 'Face Detection', 'face', TRUE, 98.50, 'Advanced facial recognition using deep learning'),
('model-person', 'Person Detection', 'person', TRUE, 96.20, 'Full body person detection and tracking'),
('model-text', 'Text Recognition', 'text', TRUE, 94.70, 'OCR and text extraction from images'),
('model-medical', 'Medical Info Detection', 'medical', TRUE, 97.10, 'Identifies medical records and health information'),
('model-license', 'License Plate Detection', 'license-plate', TRUE, 99.30, 'Vehicle license plate recognition');
