import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../../ui/atoms/buttons/btn-primary.component';
import { BtnSecondaryComponent } from '../../../ui/atoms/buttons/btn-secondary/btn-secondary';
import { BadgeComponent } from '../../../ui/atoms/misc/badge/badge.component';
import { ProgressBarComponent } from '../../../ui/atoms/feedback/progress-bar/progress-bar.component';
import { CardBasicComponent } from '../../../ui/atoms/card-basic/card-basic';

export interface ComplianceRequirement {
  id: string;
  title: string;
  description: string;
  status: 'met' | 'pending' | 'failed';
  category: 'data_protection' | 'privacy' | 'security' | 'audit';
}

export interface ComplianceStandard {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: 'primary' | 'success' | 'error' | 'warning'; // Removed 'neutral'
  requirements: ComplianceRequirement[];
  complianceLevel: number;
  lastUpdated: Date;
}

@Component({
  selector: 'app-compliance',
  standalone: true,
  imports: [CommonModule, BtnPrimaryComponent, BtnSecondaryComponent, BadgeComponent, CardBasicComponent, ProgressBarComponent],
  templateUrl: './compliance.component.html',
  styleUrls: ['./compliance.component.scss']
})
export class ComplianceComponent {
  // Signals for state management
  standards = signal<ComplianceStandard[]>([
    {
      id: 'gdpr',
      name: 'GDPR',
      description: 'General Data Protection Regulation (EU)',
      icon: '🇪🇺',
      color: 'primary',
      complianceLevel: 85,
      lastUpdated: new Date('2024-01-15'),
      requirements: [
        { id: 'gdpr-1', title: 'Data Minimization', description: 'Only collect necessary personal data', status: 'met', category: 'data_protection' },
        { id: 'gdpr-2', title: 'Consent Management', description: 'Obtain explicit user consent', status: 'met', category: 'privacy' },
        { id: 'gdpr-3', title: 'Right to Erasure', description: 'Implement data deletion requests', status: 'pending', category: 'data_protection' },
        { id: 'gdpr-4', title: 'Data Portability', description: 'Allow data export in standard format', status: 'failed', category: 'privacy' }
      ]
    },
    {
      id: 'hipaa',
      name: 'HIPAA',
      description: 'Health Insurance Portability and Accountability Act (US)',
      icon: '🇺🇸',
      color: 'success',
      complianceLevel: 70,
      lastUpdated: new Date('2024-01-10'),
      requirements: [
        { id: 'hipaa-1', title: 'PHI Protection', description: 'Protect Protected Health Information', status: 'met', category: 'security' },
        { id: 'hipaa-2', title: 'Access Controls', description: 'Role-based access to medical data', status: 'pending', category: 'security' },
        { id: 'hipaa-3', title: 'Audit Logs', description: 'Maintain access and modification logs', status: 'met', category: 'audit' }
      ]
    },
    {
      id: 'dpdp',
      name: 'DPDP',
      description: 'Digital Personal Data Protection Act (India)',
      icon: '🇮🇳',
      color: 'warning',
      complianceLevel: 60,
      lastUpdated: new Date('2024-01-05'),
      requirements: [
        { id: 'dpdp-1', title: 'Data Principal Rights', description: 'Respect individual data rights', status: 'pending', category: 'privacy' },
        { id: 'dpdp-2', title: 'Data Fiduciary Duties', description: 'Appoint data protection officer', status: 'failed', category: 'data_protection' }
      ]
    }
  ]);

  selectedStandardId = signal<string>('gdpr');
  showAllRequirements = signal<boolean>(false);

  // Computed signals - moved all complex logic here
  selectedStandard = computed(() => {
    return this.standards().find(standard => standard.id === this.selectedStandardId());
  });

  totalRequirementsCount = computed(() => {
    return this.standards().reduce((total, s) => total + s.requirements.length, 0);
  });

  metRequirementsCount = computed(() => {
    const standard = this.selectedStandard();
    if (!standard) return 0;
    return standard.requirements.filter(r => r.status === 'met').length;
  });

  standardMetRequirements = computed(() => {
    const standards = this.standards();
    const result: {[key: string]: number} = {};
    
    standards.forEach(standard => {
      result[standard.id] = standard.requirements.filter(r => r.status === 'met').length;
    });
    
    return result;
  });

  filteredRequirements = computed(() => {
    const standard = this.selectedStandard();
    if (!standard) return [];
    
    if (this.showAllRequirements()) {
      return standard.requirements;
    } else {
      return standard.requirements.filter(r => r.status !== 'met');
    }
  });

  // Methods
  selectStandard(standardId: string): void {
    this.selectedStandardId.set(standardId);
  }

  toggleRequirementsView(): void {
    this.showAllRequirements.update(value => !value);
  }

  updateRequirementStatus(requirementId: string, status: 'met' | 'pending' | 'failed'): void {
    this.standards.update(standards => 
      standards.map(standard => ({
        ...standard,
        requirements: standard.requirements.map(req =>
          req.id === requirementId ? { ...req, status } : req
        )
      }))
    );
  }

  getOverallCompliance(): number {
    const allRequirements = this.standards().flatMap(s => s.requirements);
    const metRequirements = allRequirements.filter(r => r.status === 'met');
    return allRequirements.length > 0 ? Math.round((metRequirements.length / allRequirements.length) * 100) : 0;
  }

  getRequirementBadgeVariant(status: 'met' | 'pending' | 'failed'): 'primary' | 'success' | 'error' | 'warning' {
    switch (status) {
      case 'met': return 'success';
      case 'pending': return 'warning';
      case 'failed': return 'error';
      default: return 'primary';
    }
  }
}