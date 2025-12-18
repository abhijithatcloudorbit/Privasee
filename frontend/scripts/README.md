# Privacy Shield Database Scripts

This directory contains SQL scripts to set up the complete database schema and populate it with sample data for the Privacy Shield application.

## Database Requirements

- MySQL 8.0+ or PostgreSQL 12+
- Minimum 100MB storage space
- UTF-8 character set support

## Script Execution Order

Run the scripts in the following order:

1. **01_create_schema.sql** - Creates all database tables and indexes
2. **02_seed_users.sql** - Populates users table with sample Indian users
3. **03_seed_detection_models.sql** - Adds ML detection models
4. **04_seed_uploads.sql** - Creates sample upload records
5. **05_seed_detections.sql** - Adds detection results for uploads
6. **06_seed_compliance_logs.sql** - Populates audit trail
7. **07_seed_dashboard_metrics.sql** - Seeds dashboard statistics
8. **08_seed_user_settings.sql** - Configures user preferences

## Quick Start

### MySQL
\`\`\`bash
mysql -u your_username -p your_database < scripts/01_create_schema.sql
mysql -u your_username -p your_database < scripts/02_seed_users.sql
mysql -u your_username -p your_database < scripts/03_seed_detection_models.sql
mysql -u your_username -p your_database < scripts/04_seed_uploads.sql
mysql -u your_username -p your_database < scripts/05_seed_detections.sql
mysql -u your_username -p your_database < scripts/06_seed_compliance_logs.sql
mysql -u your_username -p your_database < scripts/07_seed_dashboard_metrics.sql
mysql -u your_username -p your_database < scripts/08_seed_user_settings.sql
\`\`\`

### PostgreSQL
\`\`\`bash
psql -U your_username -d your_database -f scripts/01_create_schema.sql
psql -U your_username -d your_database -f scripts/02_seed_users.sql
psql -U your_username -d your_database -f scripts/03_seed_detection_models.sql
psql -U your_username -d your_database -f scripts/04_seed_uploads.sql
psql -U your_username -d your_database -f scripts/05_seed_detections.sql
psql -U your_username -d your_database -f scripts/06_seed_compliance_logs.sql
psql -U your_username -d your_database -f scripts/07_seed_dashboard_metrics.sql
psql -U your_username -d your_database -f scripts/08_seed_user_settings.sql
\`\`\`

## Demo Credentials

All users have the same password for testing: **demo123**

Sample users:
- **arjun.patel@privacyshield.com** - Admin role
- **priya.sharma@privacyshield.com** - Analyst role  
- **rahul.kumar@privacyshield.com** - Developer role
- **demo@privacyshield.com** - Demo admin account

## Database Schema Overview

### Core Tables
- **users** - User accounts and authentication
- **uploads** - Uploaded files and metadata
- **detections** - ML detection results with bounding boxes
- **detection_models** - Configuration for ML models

### Processing Tables
- **processing_queue** - Real-time processing status
- **model_processing_results** - Individual model results per upload

### Compliance Tables
- **compliance_logs** - Audit trail for all actions
- **compliance_reports** - Generated compliance reports

### Configuration Tables
- **user_settings** - User preferences and settings
- **api_keys** - API authentication keys
- **dashboard_metrics** - Cached metrics for performance

## Notes

- All timestamps use UTC timezone
- Password hashes use bcrypt with 10 rounds
- UUIDs are used for primary keys
- Foreign key constraints ensure referential integrity
- Indexes are optimized for common query patterns

## Security Considerations

1. Change all default passwords in production
2. Use environment variables for database credentials
3. Enable SSL/TLS for database connections
4. Regular backups with point-in-time recovery
5. Implement row-level security for multi-tenant deployments

## Data Retention

Consider implementing data retention policies:
- Compliance logs: 7 years (regulatory requirement)
- Uploads and detections: 90 days (configurable)
- Processing queue: 30 days
- Dashboard metrics: 365 days
