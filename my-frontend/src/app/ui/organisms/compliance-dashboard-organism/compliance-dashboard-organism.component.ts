import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ComplianceRequirement {
  id: string;
  standard: 'GDPR' | 'HIPAA' | 'DPDP';
  requirement: string;
  description: string;
  status: 'compliant' | 'partial' | 'non-compliant' | 'not-applicable';
  lastChecked: Date;
  nextCheck: Date;
  actions: string[];
  severity: 'high' | 'medium' | 'low';
}

@Component({
  selector: 'app-compliance-dashboard-organism',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './compliance-dashboard-organism.component.html',
  styleUrls: ['./compliance-dashboard-organism.component.scss']
})
export class ComplianceDashboardOrganismComponent {
  // Current date for the component
  currentDate = new Date();
  
  // Signals for compliance data
  complianceRequirements = signal<ComplianceRequirement[]>([
    {
      id: 'gdpr-1',
      standard: 'GDPR',
      requirement: 'Data Minimization',
      description: 'Only collect data necessary for specified purposes',
      status: 'compliant',
      lastChecked: new Date('2024-01-15'),
      nextCheck: new Date('2024-07-15'),
      actions: ['Automatic data cleanup enabled', 'Purpose limitation in place'],
      severity: 'high'
    },
    {
      id: 'gdpr-2',
      standard: 'GDPR',
      requirement: 'Right to Erasure',
      description: 'Allow users to request deletion of their data',
      status: 'partial',
      lastChecked: new Date('2024-01-10'),
      nextCheck: new Date('2024-04-10'),
      actions: ['Manual review required', '30-day deletion policy'],
      severity: 'high'
    },
    {
      id: 'hipaa-1',
      standard: 'HIPAA',
      requirement: 'PHI Protection',
      description: 'Protect Protected Health Information',
      status: 'compliant',
      lastChecked: new Date('2024-02-01'),
      nextCheck: new Date('2024-08-01'),
      actions: ['Encryption enabled', 'Access logs active'],
      severity: 'high'
    },
    {
      id: 'dpdp-1',
      standard: 'DPDP',
      requirement: 'Consent Management',
      description: 'Obtain explicit consent for data processing',
      status: 'non-compliant',
      lastChecked: new Date('2024-01-20'),
      nextCheck: new Date('2024-02-20'),
      actions: ['Implement consent forms', 'Add consent tracking'],
      severity: 'high'
    },
    {
      id: 'gdpr-3',
      standard: 'GDPR',
      requirement: 'Data Portability',
      description: 'Allow users to export their data',
      status: 'compliant',
      lastChecked: new Date('2024-01-05'),
      nextCheck: new Date('2024-07-05'),
      actions: ['JSON export available', 'CSV export in development'],
      severity: 'medium'
    }
  ]);

  // Computed statistics
  complianceStats = computed(() => {
    const requirements = this.complianceRequirements();
    const total = requirements.length;
    
    const byStandard = {
      GDPR: requirements.filter(r => r.standard === 'GDPR').length,
      HIPAA: requirements.filter(r => r.standard === 'HIPAA').length,
      DPDP: requirements.filter(r => r.standard === 'DPDP').length
    };
    
    const byStatus = {
      compliant: requirements.filter(r => r.status === 'compliant').length,
      partial: requirements.filter(r => r.status === 'partial').length,
      nonCompliant: requirements.filter(r => r.status === 'non-compliant').length,
      notApplicable: requirements.filter(r => r.status === 'not-applicable').length
    };
    
    const complianceScore = Math.round(
      ((byStatus.compliant + (byStatus.partial * 0.5)) / total) * 100
    );
    
    return { total, byStandard, byStatus, complianceScore };
  });

  // Filter signals
  filterStandard = signal<'all' | ComplianceRequirement['standard']>('all');
  filterStatus = signal<'all' | ComplianceRequirement['status']>('all');
  filterSeverity = signal<'all' | ComplianceRequirement['severity']>('all');

  // Filtered requirements
  filteredRequirements = computed(() => {
    return this.complianceRequirements().filter(req => {
      const standardMatch = this.filterStandard() === 'all' || req.standard === this.filterStandard();
      const statusMatch = this.filterStatus() === 'all' || req.status === this.filterStatus();
      const severityMatch = this.filterSeverity() === 'all' || req.severity === this.filterSeverity();
      
      return standardMatch && statusMatch && severityMatch;
    });
  });

  // Standard colors
  getStandardColor(standard: ComplianceRequirement['standard']): string {
    const colors = {
      'GDPR': '#3498db',
      'HIPAA': '#2ecc71',
      'DPDP': '#9b59b6'
    };
    return colors[standard] || '#95a5a6';
  }

  // Status colors
  getStatusColor(status: ComplianceRequirement['status']): string {
    const colors = {
      'compliant': '#2ecc71',
      'partial': '#f39c12',
      'non-compliant': '#e74c3c',
      'not-applicable': '#95a5a6'
    };
    return colors[status] || '#95a5a6';
  }

  // Severity colors
  getSeverityColor(severity: ComplianceRequirement['severity']): string {
    const colors = {
      'high': '#e74c3c',
      'medium': '#f39c12',
      'low': '#2ecc71'
    };
    return colors[severity] || '#95a5a6';
  }

  // Format date
  formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  }

  // Calculate days until next check
  daysUntilNextCheck(date: Date): number {
    const nextCheck = new Date(date);
    const diffTime = nextCheck.getTime() - this.currentDate.getTime();
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  }

  // Helper method for title case (to replace missing pipe)
  toTitleCase(str: string): string {
    return str.replace(/_/g, ' ')
             .replace(/-/g, ' ')
             .toLowerCase()
             .split(' ')
             .map(word => word.charAt(0).toUpperCase() + word.slice(1))
             .join(' ');
  }

  // Export compliance report
  exportComplianceReport(): void {
    const report = {
      generatedAt: this.currentDate.toISOString(),
      stats: this.complianceStats(),
      requirements: this.complianceRequirements()
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `compliance-report-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }
}