import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

// Interfaces defined in the component file
interface AuditEvent {
  id: string;
  timestamp: Date;
  user: string;
  action: string;
  category: 'upload' | 'processing' | 'edit' | 'export' | 'compliance' | 'system' | 'security';
  severity: 'info' | 'warning' | 'error' | 'critical';
  description: string;
  details: AuditEventDetails;
  metadata?: Record<string, any>;
}

// Flexible details interface to accommodate different event types
interface AuditEventDetails {
  imageId?: string;
  batchId?: string;
  detectionCount?: number;
  filterApplied?: string;
  complianceRule?: string;
  ipAddress?: string;
  userAgent?: string;
  durationMs?: number;
  // Additional properties used in mock data
  imageCount?: number;
  destination?: string;
  requestId?: string;
  attemptCount?: number;
  blocked?: boolean;
  userId?: string;
  consentType?: string;
  status?: string; // For consent status
  retentionPeriod?: string;
  deletionReason?: string;
  backupSize?: string;
  // For compliance audit
  requirementsChecked?: number;
  // Make it extensible for other properties
  [key: string]: any;
}

interface AuditFilter {
  category: string | 'all';
  severity: string | 'all';
  user: string | 'all';
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  searchQuery: string;
}

interface ComplianceRequirement {
  id: string;
  regulation: 'GDPR' | 'HIPAA' | 'DPDP' | 'CCPA' | 'LGPD';
  article: string;
  description: string;
  applicable: boolean;
  status: 'compliant' | 'partial' | 'non-compliant';
  lastAudit: Date;
}

@Component({
  selector: 'app-audit-trail-viewer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './audit-trail-viewer.component.html',
  styleUrls: ['./audit-trail-viewer.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AuditTrailViewerComponent {
  // Current date for the component
  currentDate = new Date();
  
  // Audit Events Data
  auditEvents = signal<AuditEvent[]>([
    {
      id: 'audit-001',
      timestamp: new Date('2024-01-26T10:30:00'),
      user: 'admin@company.com',
      action: 'Batch Upload',
      category: 'upload',
      severity: 'info',
      description: 'Uploaded 25 images for processing',
      details: {
        batchId: 'BATCH-2024-001',
        imageCount: 25,
        ipAddress: '192.168.1.100',
        userAgent: 'Chrome/120.0.0.0',
        durationMs: 2450
      }
    },
    {
      id: 'audit-002',
      timestamp: new Date('2024-01-26T10:32:15'),
      user: 'admin@company.com',
      action: 'AI Processing',
      category: 'processing',
      severity: 'info',
      description: 'AI detected 48 sensitive elements across 25 images',
      details: {
        batchId: 'BATCH-2024-001',
        detectionCount: 48,
        durationMs: 12500
      }
    },
    {
      id: 'audit-003',
      timestamp: new Date('2024-01-26T10:35:30'),
      user: 'privacy-officer@company.com',
      action: 'Manual Redaction',
      category: 'edit',
      severity: 'warning',
      description: 'Manual redaction applied to 3 additional faces',
      details: {
        imageId: 'IMG-2024-001-15',
        filterApplied: 'blur',
        detectionCount: 3
      }
    },
    {
      id: 'audit-004',
      timestamp: new Date('2024-01-26T10:40:00'),
      user: 'admin@company.com',
      action: 'GDPR Compliance Check',
      category: 'compliance',
      severity: 'info',
      description: 'GDPR Article 17 compliance verified for batch',
      details: {
        batchId: 'BATCH-2024-001',
        complianceRule: 'GDPR-17',
        durationMs: 3200
      }
    },
    {
      id: 'audit-005',
      timestamp: new Date('2024-01-26T10:42:15'),
      user: 'system',
      action: 'Export Completed',
      category: 'export',
      severity: 'info',
      description: 'Processed images exported to secure storage',
      details: {
        batchId: 'BATCH-2024-001',
        imageCount: 25,
        destination: 's3://secure-bucket/batch-2024-001'
      }
    },
    {
      id: 'audit-006',
      timestamp: new Date('2024-01-26T09:15:00'),
      user: 'analyst@company.com',
      action: 'Data Access Request',
      category: 'compliance',
      severity: 'critical',
      description: 'Subject Access Request processed under GDPR Article 15',
      details: {
        complianceRule: 'GDPR-15',
        requestId: 'SAR-2024-001',
        durationMs: 5600
      }
    },
    {
      id: 'audit-007',
      timestamp: new Date('2024-01-26T08:45:30'),
      user: 'system',
      action: 'Security Alert',
      category: 'security',
      severity: 'error',
      description: 'Multiple failed login attempts detected',
      details: {
        ipAddress: '203.0.113.25',
        attemptCount: 5,
        blocked: true
      }
    },
    {
      id: 'audit-008',
      timestamp: new Date('2024-01-25T16:20:00'),
      user: 'admin@company.com',
      action: 'Consent Record Updated',
      category: 'compliance',
      severity: 'info',
      description: 'User consent preferences updated for marketing emails',
      details: {
        userId: 'USER-04567',
        consentType: 'marketing',
        status: 'revoked'
      }
    },
    {
      id: 'audit-009',
      timestamp: new Date('2024-01-25T14:10:15'),
      user: 'processor@company.com',
      action: 'Batch Deletion',
      category: 'compliance',
      severity: 'warning',
      description: 'Batch deleted in accordance with retention policy',
      details: {
        batchId: 'BATCH-2023-987',
        retentionPeriod: '30 days',
        deletionReason: 'retention_policy'
      }
    },
    {
      id: 'audit-010',
      timestamp: new Date('2024-01-25T11:30:45'),
      user: 'system',
      action: 'System Backup',
      category: 'system',
      severity: 'info',
      description: 'Automated system backup completed successfully',
      details: {
        backupSize: '45.2 GB',
        durationMs: 1250000,
        destination: 'offsite-backup-server'
      }
    }
  ]);

  // Compliance Requirements
  complianceRequirements = signal<ComplianceRequirement[]>([
    {
      id: 'req-001',
      regulation: 'GDPR',
      article: 'Article 5',
      description: 'Principles relating to processing of personal data',
      applicable: true,
      status: 'compliant',
      lastAudit: new Date('2024-01-25')
    },
    {
      id: 'req-002',
      regulation: 'GDPR',
      article: 'Article 6',
      description: 'Lawfulness of processing',
      applicable: true,
      status: 'compliant',
      lastAudit: new Date('2024-01-25')
    },
    {
      id: 'req-003',
      regulation: 'GDPR',
      article: 'Article 15',
      description: 'Right of access by the data subject',
      applicable: true,
      status: 'compliant',
      lastAudit: new Date('2024-01-26')
    },
    {
      id: 'req-004',
      regulation: 'GDPR',
      article: 'Article 17',
      description: 'Right to erasure ("right to be forgotten")',
      applicable: true,
      status: 'partial',
      lastAudit: new Date('2024-01-24')
    },
    {
      id: 'req-005',
      regulation: 'HIPAA',
      article: '§164.308',
      description: 'Administrative safeguards',
      applicable: true,
      status: 'compliant',
      lastAudit: new Date('2024-01-23')
    },
    {
      id: 'req-006',
      regulation: 'HIPAA',
      article: '§164.312',
      description: 'Technical safeguards',
      applicable: true,
      status: 'compliant',
      lastAudit: new Date('2024-01-23')
    },
    {
      id: 'req-007',
      regulation: 'DPDP',
      article: 'Section 5',
      description: 'Notice and consent requirements',
      applicable: true,
      status: 'partial',
      lastAudit: new Date('2024-01-22')
    },
    {
      id: 'req-008',
      regulation: 'DPDP',
      article: 'Section 8',
      description: 'Rights of data principals',
      applicable: true,
      status: 'non-compliant',
      lastAudit: new Date('2024-01-21')
    },
    {
      id: 'req-009',
      regulation: 'CCPA',
      article: '1798.100',
      description: 'Notice at collection',
      applicable: false,
      status: 'compliant',
      lastAudit: new Date('2024-01-20')
    },
    {
      id: 'req-010',
      regulation: 'LGPD',
      article: 'Article 18',
      description: 'Right to deletion',
      applicable: false,
      status: 'compliant',
      lastAudit: new Date('2024-01-19')
    }
  ]);

  // Users for filtering
  users = signal<string[]>([
    'admin@company.com',
    'privacy-officer@company.com',
    'analyst@company.com',
    'processor@company.com',
    'system'
  ]);

  // Filter State
  filter = signal<AuditFilter>({
    category: 'all',
    severity: 'all',
    user: 'all',
    dateRange: {
      start: new Date(new Date().setDate(new Date().getDate() - 7)),
      end: new Date()
    },
    searchQuery: ''
  });

  // UI State Signals
  viewMode = signal<'timeline' | 'table' | 'compliance'>('timeline');
  selectedEventId = signal<string | null>(null);
  isLoading = signal(false);
  exportFormat = signal<'json' | 'csv' | 'pdf'>('json');
  
  // New signals for missing template references
  selectedRegulation = signal<string>('GDPR');

  // Computed: Filtered events
  filteredEvents = computed(() => {
    let events = this.auditEvents();
    const filter = this.filter();

    // Filter by category
    if (filter.category !== 'all') {
      events = events.filter(event => event.category === filter.category);
    }

    // Filter by severity
    if (filter.severity !== 'all') {
      events = events.filter(event => event.severity === filter.severity);
    }

    // Filter by user
    if (filter.user !== 'all') {
      events = events.filter(event => event.user === filter.user);
    }

    // Filter by date range
    if (filter.dateRange.start) {
      events = events.filter(event => event.timestamp >= filter.dateRange.start!);
    }
    if (filter.dateRange.end) {
      const endDate = new Date(filter.dateRange.end);
      endDate.setHours(23, 59, 59, 999);
      events = events.filter(event => event.timestamp <= endDate);
    }

    // Filter by search query
    if (filter.searchQuery.trim()) {
      const query = filter.searchQuery.toLowerCase();
      events = events.filter(event => 
        event.action.toLowerCase().includes(query) ||
        event.description.toLowerCase().includes(query) ||
        event.user.toLowerCase().includes(query)
      );
    }

    return events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  });

  // Computed: Filtered compliance requirements by selected regulation
  filteredComplianceRequirements = computed(() => {
    const selected = this.selectedRegulation();
    return this.complianceRequirements().filter(req => 
      selected === 'all' || req.regulation === selected
    );
  });

  // Computed: Statistics
  auditStats = computed(() => {
    const events = this.auditEvents();
    const filtered = this.filteredEvents();
    
    return {
      totalEvents: events.length,
      filteredEvents: filtered.length,
      todayEvents: events.filter(e => 
        e.timestamp.toDateString() === new Date().toDateString()
      ).length,
      criticalEvents: events.filter(e => e.severity === 'critical').length,
      uniqueUsers: new Set(events.map(e => e.user)).size,
      complianceEvents: events.filter(e => e.category === 'compliance').length
    };
  });

  // Computed: Compliance summary
  complianceSummary = computed(() => {
    const requirements = this.complianceRequirements();
    
    return {
      total: requirements.length,
      applicable: requirements.filter(r => r.applicable).length,
      compliant: requirements.filter(r => r.applicable && r.status === 'compliant').length,
      partial: requirements.filter(r => r.applicable && r.status === 'partial').length,
      nonCompliant: requirements.filter(r => r.applicable && r.status === 'non-compliant').length,
      byRegulation: {
        GDPR: requirements.filter(r => r.regulation === 'GDPR').length,
        HIPAA: requirements.filter(r => r.regulation === 'HIPAA').length,
        DPDP: requirements.filter(r => r.regulation === 'DPDP').length,
        CCPA: requirements.filter(r => r.regulation === 'CCPA').length,
        LGPD: requirements.filter(r => r.regulation === 'LGPD').length
      }
    };
  });

  // Computed: Event categories count
  categoryCounts = computed(() => {
    const events = this.auditEvents();
    const counts: Record<string, number> = {};
    
    events.forEach(event => {
      counts[event.category] = (counts[event.category] || 0) + 1;
    });
    
    return counts;
  });

  // Computed: Event severity count
  severityCounts = computed(() => {
    const events = this.auditEvents();
    const counts: Record<string, number> = {};
    
    events.forEach(event => {
      counts[event.severity] = (counts[event.severity] || 0) + 1;
    });
    
    return counts;
  });

  // Computed: Selected event
  selectedEvent = computed(() => {
    const selectedId = this.selectedEventId();
    if (!selectedId) return null;
    
    return this.auditEvents().find(event => event.id === selectedId) || null;
  });

  // Helper Methods
  formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  formatRelativeTime(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays < 7) {
      return `${diffDays}d ago`;
    } else {
      return this.formatDate(date);
    }
  }

  getSeverityColor(severity: string): string {
    const colors: Record<string, string> = {
      'info': '#3498db',
      'warning': '#f39c12',
      'error': '#e74c3c',
      'critical': '#c0392b'
    };
    return colors[severity] || '#95a5a6';
  }

  getSeverityIcon(severity: string): string {
    const icons: Record<string, string> = {
      'info': 'ℹ️',
      'warning': '⚠️',
      'error': '❌',
      'critical': '🚨'
    };
    return icons[severity] || '📝';
  }

  getCategoryIcon(category: string): string {
    const icons: Record<string, string> = {
      'upload': '📤',
      'processing': '⚙️',
      'edit': '✏️',
      'export': '📥',
      'compliance': '📋',
      'system': '🖥️',
      'security': '🔒'
    };
    return icons[category] || '📊';
  }

  getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      'upload': '#3498db',
      'processing': '#9b59b6',
      'edit': '#2ecc71',
      'export': '#e67e22',
      'compliance': '#1abc9c',
      'system': '#34495e',
      'security': '#e74c3c'
    };
    return colors[category] || '#95a5a6';
  }

  getComplianceColor(status: string): string {
    const colors: Record<string, string> = {
      'compliant': '#2ecc71',
      'partial': '#f39c12',
      'non-compliant': '#e74c3c'
    };
    return colors[status] || '#95a5a6';
  }

  getComplianceIcon(status: string): string {
    const icons: Record<string, string> = {
      'compliant': '✅',
      'partial': '⚠️',
      'non-compliant': '❌'
    };
    return icons[status] || '❓';
  }

  getRegulationColor(regulation: string): string {
    const colors: Record<string, string> = {
      'GDPR': '#3498db',
      'HIPAA': '#2ecc71',
      'DPDP': '#e74c3c',
      'CCPA': '#f39c12',
      'LGPD': '#9b59b6'
    };
    return colors[regulation] || '#95a5a6';
  }

  // NEW METHODS FOR TEMPLATE ERRORS
  getEventDetailsArray(event: AuditEvent): Array<{key: string, value: any}> {
    if (!event.details) return [];
    return Object.entries(event.details).map(([key, value]) => ({ key, value }));
  }

  getRelatedAuditEvents(requirement: ComplianceRequirement): AuditEvent[] {
    return this.auditEvents().filter(event => {
      return event.category === 'compliance' && 
             event.details?.complianceRule?.includes(requirement.regulation);
    });
  }

  refreshData(): void {
    // In a real app, this would fetch fresh data from the server
    console.log('Refreshing audit data...');
    this.isLoading.set(true);
    
    setTimeout(() => {
      this.isLoading.set(false);
      
      // Add a refresh audit event
      const refreshEvent: AuditEvent = {
        id: `audit-${Date.now()}`,
        timestamp: new Date(),
        user: 'system',
        action: 'Data Refresh',
        category: 'system',
        severity: 'info',
        description: 'Audit trail data refreshed manually',
        details: {
          durationMs: 500,
          refreshedAt: new Date().toISOString()
        }
      };
      
      this.auditEvents.update(events => [refreshEvent, ...events]);
    }, 1000);
  }

  // Helper for safe date formatting in template
  formatDateForInput(date: Date | null): string {
    if (!date) return '';
    return date.toISOString().split('T')[0];
  }

  // Helper for safe regulation count access
  getRegulationCount(regulation: string): number {
    const summary = this.complianceSummary();
    const byRegulation = summary.byRegulation as Record<string, number>;
    return byRegulation[regulation] || 0;
  }

  // Methods for date range changes
  updateStartDate(event: Event): void {
    const input = event.target as HTMLInputElement;
    const newDate = input.value ? new Date(input.value) : null;
    this.filter.update(f => ({
      ...f,
      dateRange: {
        ...f.dateRange,
        start: newDate
      }
    }));
  }

  updateEndDate(event: Event): void {
    const input = event.target as HTMLInputElement;
    const newDate = input.value ? new Date(input.value) : null;
    this.filter.update(f => ({
      ...f,
      dateRange: {
        ...f.dateRange,
        end: newDate
      }
    }));
  }

  // Safe method to check if details exist
  hasDetails(details: any): boolean {
    if (!details) return false;
    return Object.keys(details).length > 0;
  }

  // Filter Methods
  setCategoryFilter(category: string): void {
    this.filter.update(f => ({ ...f, category }));
  }

  setSeverityFilter(severity: string): void {
    this.filter.update(f => ({ ...f, severity }));
  }

  setUserFilter(user: string): void {
    this.filter.update(f => ({ ...f, user }));
  }

  setDateRange(start: Date | null, end: Date | null): void {
    this.filter.update(f => ({ 
      ...f, 
      dateRange: { start, end } 
    }));
  }

  setSearchQuery(query: string): void {
    this.filter.update(f => ({ ...f, searchQuery: query }));
  }

  resetFilters(): void {
    this.filter.set({
      category: 'all',
      severity: 'all',
      user: 'all',
      dateRange: {
        start: new Date(new Date().setDate(new Date().getDate() - 7)),
        end: new Date()
      },
      searchQuery: ''
    });
  }

  // View Mode Methods
  setViewMode(mode: 'timeline' | 'table' | 'compliance'): void {
    this.viewMode.set(mode);
  }

  selectEvent(eventId: string): void {
    this.selectedEventId.set(eventId);
  }

  clearSelection(): void {
    this.selectedEventId.set(null);
  }

  // Data Generation Methods
  generateTestEvent(): void {
    const categories: AuditEvent['category'][] = ['upload', 'processing', 'edit', 'export', 'compliance', 'system', 'security'];
    const severities: AuditEvent['severity'][] = ['info', 'warning', 'error', 'critical'];
    const users = this.users();
    
    const newEvent: AuditEvent = {
      id: `audit-${Date.now()}`,
      timestamp: new Date(),
      user: users[Math.floor(Math.random() * users.length)],
      action: `Test Action ${Math.floor(Math.random() * 100)}`,
      category: categories[Math.floor(Math.random() * categories.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      description: 'This is a test audit event generated for demonstration purposes',
      details: {
        durationMs: Math.floor(Math.random() * 5000),
        ipAddress: `192.168.1.${Math.floor(Math.random() * 255)}`
      }
    };
    
    this.auditEvents.update(events => [newEvent, ...events]);
  }

  // Export Methods
  exportAuditData(): void {
    const events = this.filteredEvents();
    const format = this.exportFormat();
    
    switch (format) {
      case 'json':
        this.exportAsJson(events);
        break;
      case 'csv':
        this.exportAsCsv(events);
        break;
      case 'pdf':
        this.exportAsPdf(events);
        break;
    }
  }

  private exportAsJson(events: AuditEvent[]): void {
    const exportData = {
      exportDate: this.currentDate.toISOString(),
      filter: this.filter(),
      eventCount: events.length,
      events: events.map(event => ({
        ...event,
        timestamp: event.timestamp.toISOString()
      }))
    };
    
    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `audit-trail-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }

  private exportAsCsv(events: AuditEvent[]): void {
    const headers = ['Timestamp', 'User', 'Action', 'Category', 'Severity', 'Description'];
    const csvRows = events.map(event => [
      event.timestamp.toISOString(),
      event.user,
      event.action,
      event.category,
      event.severity,
      event.description
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvRows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `audit-trail-${this.currentDate.toISOString().slice(0, 10)}.csv`;
    link.click();
  }

  private exportAsPdf(events: AuditEvent[]): void {
    // In a real app, this would use a PDF generation library
    console.log('PDF export would be implemented with a library like jsPDF');
    alert('PDF export requires additional libraries. Using JSON export instead.');
    this.exportFormat.set('json');
    this.exportAsJson(events);
  }

  // Compliance Methods
  runComplianceAudit(): void {
    this.isLoading.set(true);
    
    // Simulate audit process
    setTimeout(() => {
      this.complianceRequirements.update(requirements =>
        requirements.map(req => ({
          ...req,
          lastAudit: new Date(),
          status: req.applicable ? 
            (Math.random() > 0.3 ? 'compliant' : Math.random() > 0.5 ? 'partial' : 'non-compliant') 
            : req.status
        }))
      );
      
      // Add audit event
      const auditEvent: AuditEvent = {
        id: `audit-${Date.now()}`,
        timestamp: new Date(),
        user: 'system',
        action: 'Compliance Audit',
        category: 'compliance',
        severity: 'info',
        description: 'Automated compliance audit completed',
        details: {
          durationMs: 4500,
          requirementsChecked: this.complianceRequirements().length
        }
      };
      
      this.auditEvents.update(events => [auditEvent, ...events]);
      this.isLoading.set(false);
    }, 2000);
  }

  // Search Methods
  quickSearch(query: string): void {
    this.setSearchQuery(query);
    this.setViewMode('table');
  }
}