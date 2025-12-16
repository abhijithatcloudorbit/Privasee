-- Privacy Shield Database Schema
-- Creates all necessary tables for the Privacy & Image Anonymization System

-- Users table for authentication and user management
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    department VARCHAR(100),
    location VARCHAR(100),
    avatar VARCHAR(10),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_status (status)
);

-- Uploads table for tracking uploaded files
CREATE TABLE IF NOT EXISTS uploads (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    file_type VARCHAR(100),
    thumbnail_url TEXT,
    original_url TEXT,
    processed_url TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    compliance_mode VARCHAR(50) NOT NULL,
    total_detections INT DEFAULT 0,
    processing_time_ms INT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_uploaded_at (uploaded_at)
);

-- Detection models configuration
CREATE TABLE IF NOT EXISTS detection_models (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    accuracy_score DECIMAL(5,2),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_type (type),
    INDEX idx_enabled (enabled)
);

-- Detections table for storing detected sensitive content
CREATE TABLE IF NOT EXISTS detections (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    upload_id VARCHAR(36) NOT NULL,
    model_id VARCHAR(36) NOT NULL,
    detection_type VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,2) NOT NULL,
    bounding_box_x DECIMAL(5,2),
    bounding_box_y DECIMAL(5,2),
    bounding_box_width DECIMAL(5,2),
    bounding_box_height DECIMAL(5,2),
    status VARCHAR(50) DEFAULT 'pending',
    blur_applied BOOLEAN DEFAULT FALSE,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES detection_models(id) ON DELETE CASCADE,
    INDEX idx_upload_id (upload_id),
    INDEX idx_detection_type (detection_type),
    INDEX idx_confidence (confidence_score)
);

-- Processing queue for tracking real-time processing
CREATE TABLE IF NOT EXISTS processing_queue (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    upload_id VARCHAR(36) NOT NULL,
    current_model VARCHAR(100),
    progress INT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'queued',
    estimated_time_seconds INT,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE,
    INDEX idx_upload_id (upload_id),
    INDEX idx_status (status)
);

-- Model processing results for each upload
CREATE TABLE IF NOT EXISTS model_processing_results (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    processing_queue_id VARCHAR(36) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    confidence_score DECIMAL(5,2),
    detections_count INT DEFAULT 0,
    processing_time_ms INT,
    completed_at TIMESTAMP,
    FOREIGN KEY (processing_queue_id) REFERENCES processing_queue(id) ON DELETE CASCADE,
    INDEX idx_queue_id (processing_queue_id),
    INDEX idx_status (status)
);

-- Compliance logs for audit trail
CREATE TABLE IF NOT EXISTS compliance_logs (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    upload_id VARCHAR(36),
    user_id VARCHAR(36) NOT NULL,
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    compliance_standard VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_upload_id (upload_id),
    INDEX idx_action (action),
    INDEX idx_created_at (created_at)
);

-- Compliance reports
CREATE TABLE IF NOT EXISTS compliance_reports (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    standard VARCHAR(50) NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    total_uploads INT DEFAULT 0,
    total_detections INT DEFAULT 0,
    compliance_score DECIMAL(5,2),
    file_url TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_standard (standard),
    INDEX idx_generated_at (generated_at)
);

-- User settings
CREATE TABLE IF NOT EXISTS user_settings (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36) UNIQUE NOT NULL,
    notifications_enabled BOOLEAN DEFAULT TRUE,
    email_alerts BOOLEAN DEFAULT TRUE,
    default_compliance_mode VARCHAR(50) DEFAULT 'strict',
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    theme VARCHAR(20) DEFAULT 'dark',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- API keys for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36) NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    permissions JSON,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_api_key (api_key),
    INDEX idx_status (status)
);

-- Dashboard metrics cache for performance
CREATE TABLE IF NOT EXISTS dashboard_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id VARCHAR(36),
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15,2) NOT NULL,
    metadata JSON,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_metric_type (metric_type),
    INDEX idx_calculated_at (calculated_at)
);
