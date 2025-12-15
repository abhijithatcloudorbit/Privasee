// consent-manager-organism.component.ts
import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Interfaces
interface ConsentForm {
  id: string;
  title: string;
  description: string;
  regulation: 'GDPR' | 'HIPAA' | 'DPDP' | 'CCPA' | 'LGPD' | 'Multiple';
  version: string;
  effectiveDate: Date;
  status: 'draft' | 'active' | 'archived' | 'expired';
  required: boolean;
  sections: ConsentSection[];
  languages: string[];
  defaultLanguage: string;
  metadata: {
    createdBy: string;
    createdAt: Date;
    updatedBy: string;
    updatedAt: Date;
    approvalStatus: 'pending' | 'approved' | 'rejected';
    approvedBy?: string;
    approvedAt?: Date;
  };
}

interface ConsentSection {
  id: string;
  title: string;
  description: string;
  consentType: 'explicit' | 'implicit' | 'opt-in' | 'opt-out';
  required: boolean;
  defaultChecked: boolean;
  options: ConsentOption[];
  legalBasis: 'consent' | 'contract' | 'legal_obligation' | 'vital_interest' | 'public_task' | 'legitimate_interest';
  regulationReferences: string[];
}

interface ConsentOption {
  id: string;
  label: string;
  description: string;
  value: string;
  required: boolean;
  defaultChecked: boolean;
  category: 'marketing' | 'analytics' | 'necessary' | 'personalization' | 'third_party' | 'data_sharing';
  dataCategories: string[];
  retentionPeriod: string;
}

interface ConsentRecord {
  id: string;
  userId: string;
  userEmail: string;
  formId: string;
  formVersion: string;
  timestamp: Date;
  action: 'granted' | 'revoked' | 'updated' | 'withdrawn';
  preferences: ConsentPreference[];
  ipAddress?: string;
  userAgent?: string;
  consentMethod: 'web_form' | 'api' | 'mobile_app' | 'email' | 'in_person';
  metadata?: Record<string, any>;
}

interface ConsentPreference {
  sectionId: string;
  optionId: string;
  granted: boolean;
  timestamp: Date;
}

interface ConsentDashboardStats {
  totalForms: number;
  activeForms: number;
  totalConsents: number;
  todayConsents: number;
  pendingWithdrawals: number;
  complianceScore: number;
  byRegulation: {
    GDPR: number;
    HIPAA: number;
    DPDP: number;
    CCPA: number;
    LGPD: number;
    Multiple: number;
  };
  byCategory: Record<string, number>;
}

interface UserConsentProfile {
  userId: string;
  email: string;
  firstName?: string;
  lastName?: string;
  lastConsentDate: Date;
  activeConsents: number;
  revokedConsents: number;
  preferences: UserConsentPreference[];
  consentHistory: ConsentRecord[];
  metadata: {
    dataSubjectCategory: 'child' | 'adult' | 'vulnerable' | 'employee' | 'customer' | 'visitor' | 'patient';
    jurisdiction: string;
    ageVerified: boolean;
    parentConsent?: string;
  };
}

interface UserConsentPreference {
  formId: string;
  formTitle: string;
  granted: boolean;
  lastUpdated: Date;
  preferences: {
    [sectionId: string]: boolean;
  };
}

interface ConsentManagerFilter {
  regulation: string;
  status: string;
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  searchQuery: string;
  consentType: string;
}

@Component({
  selector: 'app-consent-manager-organism',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './consent-manager-organism.component.html',
  styleUrls: ['./consent-manager-organism.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ConsentManagerOrganismComponent {
  // Current date for the component
  currentDate = new Date();
  
  // UI State Signals
  activeView = signal<'dashboard' | 'forms' | 'users' | 'records' | 'analytics' | 'settings'>('dashboard');
  selectedFormId = signal<string | null>(null);
  selectedUserId = signal<string | null>(null);
  selectedRecordId = signal<string | null>(null);
  isLoading = signal(false);
  showConsentFormModal = signal(false);
  showWithdrawalModal = signal(false);
  selectedForm = signal<ConsentForm | null>(null);
  formMode = signal<'create' | 'edit' | 'preview'>('create');
  
  // Filter State
  filter = signal<ConsentManagerFilter>({
    regulation: 'all',
    status: 'active',
    dateRange: {
      start: new Date(new Date().setMonth(new Date().getMonth() - 1)),
      end: new Date()
    },
    searchQuery: '',
    consentType: 'all'
  });
  
  // Consent Forms Data
  consentForms = signal<ConsentForm[]>([/* Your existing form data here */]);
  
  // Consent Records Data
  consentRecords = signal<ConsentRecord[]>([/* Your existing record data here */]);
  
  // User Consent Profiles
  userProfiles = signal<UserConsentProfile[]>([/* Your existing user data here */]);

  // ========== MISSING METHODS TO FIX ERRORS ==========
  
  /**
   * Get the count of forms for a specific regulation
   */
  getRegulationCount(regulation: string): number {
    const forms = this.consentForms();
    
    if (regulation === 'Multiple') {
      return forms.filter(form => form.regulation === 'Multiple').length;
    }
    
    // For single regulations
    return forms.filter(form => form.regulation === regulation).length;
  }

  /**
   * Get the total number of consent categories with active consents
   */
  getCategoryCount(): number {
    const categoryStats = this.dashboardStats().byCategory;
    return Object.keys(categoryStats).filter(category => 
      categoryStats[category] > 0
    ).length;
  }

  /**
   * Get the percentage coverage for a regulation (forms count / total forms)
   */
  getRegulationCoveragePercent(regulation: string): string {
    const totalForms = this.dashboardStats().totalForms;
    if (totalForms === 0) return '0%';
    
    const regulationCount = this.getRegulationCount(regulation);
    const percentage = (regulationCount / totalForms) * 100;
    
    // Return as CSS width value (0-100%)
    return `${Math.min(percentage, 100)}%`;
  }

  // ========== COMPUTED SIGNALS ==========
  
  // Computed: Filtered consent forms
  filteredForms = computed(() => {
    let forms = this.consentForms();
    const filter = this.filter();
    
    // Filter by regulation
    if (filter.regulation !== 'all') {
      forms = forms.filter(form => form.regulation === filter.regulation);
    }
    
    // Filter by status
    if (filter.status !== 'all') {
      forms = forms.filter(form => form.status === filter.status);
    }
    
    // Filter by search query
    if (filter.searchQuery.trim()) {
      const query = filter.searchQuery.toLowerCase();
      forms = forms.filter(form => 
        form.title.toLowerCase().includes(query) ||
        form.description.toLowerCase().includes(query) ||
        form.regulation.toLowerCase().includes(query)
      );
    }
    
    // Filter by date range
    if (filter.dateRange.start) {
      forms = forms.filter(form => form.effectiveDate >= filter.dateRange.start!);
    }
    if (filter.dateRange.end) {
      const endDate = new Date(filter.dateRange.end);
      endDate.setHours(23, 59, 59, 999);
      forms = forms.filter(form => form.effectiveDate <= endDate);
    }
    
    return forms.sort((a, b) => b.effectiveDate.getTime() - a.effectiveDate.getTime());
  });
  
  // Computed: Filtered consent records
  filteredRecords = computed(() => {
    let records = this.consentRecords();
    const filter = this.filter();
    
    // Filter by consent type
    if (filter.consentType !== 'all') {
      // This would need to be enhanced based on actual data structure
    }
    
    // Filter by search query
    if (filter.searchQuery.trim()) {
      const query = filter.searchQuery.toLowerCase();
      records = records.filter(record => 
        record.userEmail.toLowerCase().includes(query) ||
        record.action.toLowerCase().includes(query) ||
        record.formId.toLowerCase().includes(query)
      );
    }
    
    // Filter by date range
    if (filter.dateRange.start) {
      records = records.filter(record => record.timestamp >= filter.dateRange.start!);
    }
    if (filter.dateRange.end) {
      const endDate = new Date(filter.dateRange.end);
      endDate.setHours(23, 59, 59, 999);
      records = records.filter(record => record.timestamp <= endDate);
    }
    
    return records.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  });
  
  // Computed: Dashboard statistics
  dashboardStats = computed((): ConsentDashboardStats => {
    const forms = this.consentForms();
    const records = this.consentRecords();
    const users = this.userProfiles();
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const todayConsents = records.filter(record => {
      const recordDate = new Date(record.timestamp);
      recordDate.setHours(0, 0, 0, 0);
      return recordDate.getTime() === today.getTime();
    }).length;
    
    // Calculate consent by category
    const categoryCounts: Record<string, number> = {};
    records.forEach(record => {
      record.preferences.forEach(pref => {
        const form = forms.find(f => f.id === record.formId);
        if (form) {
          form.sections.forEach(section => {
            const option = section.options.find(opt => opt.id === pref.optionId);
            if (option && pref.granted) {
              categoryCounts[option.category] = (categoryCounts[option.category] || 0) + 1;
            }
          });
        }
      });
    });
    
    // Calculate regulation counts including Multiple
    const regulationCounts = {
      GDPR: forms.filter(f => f.regulation === 'GDPR').length,
      HIPAA: forms.filter(f => f.regulation === 'HIPAA').length,
      DPDP: forms.filter(f => f.regulation === 'DPDP').length,
      CCPA: forms.filter(f => f.regulation === 'CCPA').length,
      LGPD: forms.filter(f => f.regulation === 'LGPD').length,
      Multiple: forms.filter(f => f.regulation === 'Multiple').length
    };
    
    // Calculate compliance score (simplified)
    const activeForms = forms.filter(f => f.status === 'active').length;
    const totalPossibleConsents = users.length * activeForms;
    const complianceScore = totalPossibleConsents > 0 ? 
      Math.min(100, Math.floor((records.filter(r => r.action === 'granted').length / totalPossibleConsents) * 100)) : 0;
    
    return {
      totalForms: forms.length,
      activeForms,
      totalConsents: records.length,
      todayConsents,
      pendingWithdrawals: records.filter(r => r.action === 'revoked' || r.action === 'withdrawn').length,
      complianceScore,
      byRegulation: regulationCounts,
      byCategory: categoryCounts
    };
  });
  
  // Computed: Selected form
  selectedFormDetails = computed(() => {
    const formId = this.selectedFormId();
    if (!formId) return null;
    
    return this.consentForms().find(form => form.id === formId) || null;
  });
  
  // Computed: Selected user
  selectedUserDetails = computed(() => {
    const userId = this.selectedUserId();
    if (!userId) return null;
    
    return this.userProfiles().find(user => user.userId === userId) || null;
  });
  
  // Computed: Selected record
  selectedRecordDetails = computed(() => {
    const recordId = this.selectedRecordId();
    if (!recordId) return null;
    
    return this.consentRecords().find(record => record.id === recordId) || null;
  });
  
  // Computed: Consent records for selected user
  userConsentRecords = computed(() => {
    const userId = this.selectedUserId();
    if (!userId) return [];
    
    return this.consentRecords().filter(record => record.userId === userId);
  });
  
  // Computed: Forms needing user consent (for selected user)
  formsNeedingConsent = computed(() => {
    const userId = this.selectedUserId();
    if (!userId) return [];
    
    const user = this.userProfiles().find(u => u.userId === userId);
    const activeForms = this.consentForms().filter(f => f.status === 'active' && f.required);
    
    if (!user) return activeForms;
    
    // Return forms that the user hasn't consented to yet
    return activeForms.filter(form => {
      const userConsent = user.preferences.find(p => p.formId === form.id);
      return !userConsent || !userConsent.granted;
    });
  });

  // ========== HELPER METHODS ==========
  
  formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
  
  formatDateForInput(date: Date | null): string {
    if (!date) return '';
    return date.toISOString().split('T')[0];
  }
  
  getRegulationColor(regulation: string): string {
    const colors: Record<string, string> = {
      'GDPR': '#3498db',
      'HIPAA': '#2ecc71',
      'DPDP': '#e74c3c',
      'CCPA': '#f39c12',
      'LGPD': '#9b59b6',
      'Multiple': '#34495e'
    };
    return colors[regulation] || '#95a5a6';
  }
  
  getRegulationIcon(regulation: string): string {
    const icons: Record<string, string> = {
      'GDPR': '🇪🇺',
      'HIPAA': '🏥',
      'DPDP': '🇮🇳',
      'CCPA': '🇺🇸',
      'LGPD': '🇧🇷',
      'Multiple': '🌐'
    };
    return icons[regulation] || '📋';
  }
  
  getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      'active': '#2ecc71',
      'draft': '#f39c12',
      'archived': '#95a5a6',
      'expired': '#e74c3c'
    };
    return colors[status] || '#95a5a6';
  }
  
  getConsentTypeIcon(consentType: string): string {
    const icons: Record<string, string> = {
      'explicit': '✅',
      'implicit': '🔘',
      'opt-in': '📥',
      'opt-out': '📤'
    };
    return icons[consentType] || '❓';
  }
  
  getCategoryIcon(category: string): string {
    const icons: Record<string, string> = {
      'marketing': '📢',
      'analytics': '📊',
      'necessary': '🔒',
      'personalization': '🎯',
      'third_party': '🤝',
      'data_sharing': '📤'
    };
    return icons[category] || '📝';
  }
  
  getCategoryColor(category: string): string {
    const colors: Record<string, string> = {
      'marketing': '#3498db',
      'analytics': '#9b59b6',
      'necessary': '#2ecc71',
      'personalization': '#e67e22',
      'third_party': '#f39c12',
      'data_sharing': '#e74c3c'
    };
    return colors[category] || '#95a5a6';
  }
  
  getActionIcon(action: string): string {
    const icons: Record<string, string> = {
      'granted': '✅',
      'revoked': '❌',
      'updated': '🔄',
      'withdrawn': '↩️'
    };
    return icons[action] || '📝';
  }
  
  getActionColor(action: string): string {
    const colors: Record<string, string> = {
      'granted': '#2ecc71',
      'revoked': '#e74c3c',
      'updated': '#3498db',
      'withdrawn': '#f39c12'
    };
    return colors[action] || '#95a5a6';
  }
  
  getComplianceScoreColor(score: number): string {
    if (score >= 90) return '#2ecc71';
    if (score >= 70) return '#f39c12';
    return '#e74c3c';
  }

  // ========== VIEW METHODS ==========
  
  setActiveView(view: 'dashboard' | 'forms' | 'users' | 'records' | 'analytics' | 'settings'): void {
    this.activeView.set(view);
    this.clearSelection();
  }
  
  selectForm(formId: string): void {
    this.selectedFormId.set(formId);
    this.showConsentFormModal.set(true);
  }
  
  selectUser(userId: string): void {
    this.selectedUserId.set(userId);
  }
  
  selectRecord(recordId: string): void {
    this.selectedRecordId.set(recordId);
  }
  
  clearSelection(): void {
    this.selectedFormId.set(null);
    this.selectedUserId.set(null);
    this.selectedRecordId.set(null);
    this.showConsentFormModal.set(false);
    this.showWithdrawalModal.set(false);
  }

  // ========== FORM MANAGEMENT METHODS ==========
  
  createNewForm(): void {
    const newForm: ConsentForm = {
      id: `form-${Date.now()}`,
      title: 'New Consent Form',
      description: 'Consent form description',
      regulation: 'GDPR',
      version: '1.0',
      effectiveDate: new Date(),
      status: 'draft',
      required: true,
      languages: ['en'],
      defaultLanguage: 'en',
      sections: [
        {
          id: `sec-${Date.now()}`,
          title: 'New Section',
          description: 'Section description',
          consentType: 'explicit',
          required: false,
          defaultChecked: false,
          legalBasis: 'consent',
          regulationReferences: [],
          options: [
            {
              id: `opt-${Date.now()}`,
              label: 'New Option',
              description: 'Option description',
              value: 'new_option',
              required: false,
              defaultChecked: false,
              category: 'marketing',
              dataCategories: [],
              retentionPeriod: '1 year'
            }
          ]
        }
      ],
      metadata: {
        createdBy: 'system',
        createdAt: new Date(),
        updatedBy: 'system',
        updatedAt: new Date(),
        approvalStatus: 'pending'
      }
    };
    
    this.consentForms.update(forms => [...forms, newForm]);
    this.selectedFormId.set(newForm.id);
    this.formMode.set('edit');
    this.showConsentFormModal.set(true);
    this.setActiveView('forms');
  }
  
  duplicateForm(formId: string): void {
    const originalForm = this.consentForms().find(f => f.id === formId);
    if (!originalForm) return;
    
    const duplicatedForm: ConsentForm = {
      ...originalForm,
      id: `form-${Date.now()}`,
      title: `${originalForm.title} (Copy)`,
      version: '1.0',
      effectiveDate: new Date(),
      status: 'draft',
      metadata: {
        createdBy: 'system',
        createdAt: new Date(),
        updatedBy: 'system',
        updatedAt: new Date(),
        approvalStatus: 'pending'
      }
    };
    
    this.consentForms.update(forms => [...forms, duplicatedForm]);
    this.selectedFormId.set(duplicatedForm.id);
    this.formMode.set('edit');
    this.showConsentFormModal.set(true);
  }
  
  archiveForm(formId: string): void {
    this.consentForms.update(forms =>
      forms.map(form =>
        form.id === formId ? { ...form, status: 'archived' } : form
      )
    );
  }
  
  activateForm(formId: string): void {
    this.consentForms.update(forms =>
      forms.map(form =>
        form.id === formId ? { ...form, status: 'active' } : form
      )
    );
  }

  // ========== CONSENT MANAGEMENT METHODS ==========
  
  simulateConsentGrant(userId: string, formId: string, preferences: ConsentPreference[]): void {
    const user = this.userProfiles().find(u => u.userId === userId);
    const form = this.consentForms().find(f => f.id === formId);
    
    if (!user || !form) return;
    
    const newRecord: ConsentRecord = {
      id: `record-${Date.now()}`,
      userId,
      userEmail: user.email,
      formId,
      formVersion: form.version,
      timestamp: new Date(),
      action: 'granted',
      consentMethod: 'web_form',
      ipAddress: '192.168.1.1',
      userAgent: 'Simulated/1.0',
      preferences
    };
    
    // Update user profiles
    this.userProfiles.update(profiles =>
      profiles.map(profile =>
        profile.userId === userId
          ? this.updateUserProfileOnGrant(profile, form, preferences, newRecord)
          : profile
      )
    );
    
    // Add consent record
    this.consentRecords.update(records => [...records, newRecord]);
  }
  
  private updateUserProfileOnGrant(
    profile: UserConsentProfile,
    form: ConsentForm,
    preferences: ConsentPreference[],
    record: ConsentRecord
  ): UserConsentProfile {
    const existingPreferenceIndex = profile.preferences.findIndex(p => p.formId === form.id);
    const preferencesMap = preferences.reduce((acc, pref) => {
      acc[pref.sectionId] = pref.granted;
      return acc;
    }, {} as Record<string, boolean>);
    
    const updatedPreferences = existingPreferenceIndex >= 0
      ? profile.preferences.map((p, index) =>
          index === existingPreferenceIndex
            ? {
                ...p,
                granted: true,
                lastUpdated: new Date(),
                preferences: { ...p.preferences, ...preferencesMap }
              }
            : p
        )
      : [
          ...profile.preferences,
          {
            formId: form.id,
            formTitle: form.title,
            granted: true,
            lastUpdated: new Date(),
            preferences: preferencesMap
          }
        ];
    
    return {
      ...profile,
      lastConsentDate: new Date(),
      activeConsents: profile.activeConsents + (existingPreferenceIndex < 0 ? 1 : 0),
      preferences: updatedPreferences,
      consentHistory: [...profile.consentHistory, record]
    };
  }
  
  simulateConsentRevocation(userId: string, formId: string): void {
    const user = this.userProfiles().find(u => u.userId === userId);
    const form = this.consentForms().find(f => f.id === formId);
    
    if (!user || !form) return;
    
    const newRecord: ConsentRecord = {
      id: `record-${Date.now()}`,
      userId,
      userEmail: user.email,
      formId,
      formVersion: form.version,
      timestamp: new Date(),
      action: 'revoked',
      consentMethod: 'web_form',
      ipAddress: '192.168.1.1',
      userAgent: 'Simulated/1.0',
      preferences: []
    };
    
    // Update user profiles
    this.userProfiles.update(profiles =>
      profiles.map(profile =>
        profile.userId === userId
          ? this.updateUserProfileOnRevocation(profile, formId, newRecord)
          : profile
      )
    );
    
    // Add consent record
    this.consentRecords.update(records => [...records, newRecord]);
  }
  
  private updateUserProfileOnRevocation(
    profile: UserConsentProfile,
    formId: string,
    record: ConsentRecord
  ): UserConsentProfile {
    const hadGrantedConsent = profile.preferences.some(p => p.formId === formId && p.granted);
    
    return {
      ...profile,
      lastConsentDate: new Date(),
      activeConsents: Math.max(0, hadGrantedConsent ? profile.activeConsents - 1 : profile.activeConsents),
      revokedConsents: profile.revokedConsents + 1,
      preferences: profile.preferences.map(p =>
        p.formId === formId
          ? { ...p, granted: false, lastUpdated: new Date() }
          : p
      ),
      consentHistory: [...profile.consentHistory, record]
    };
  }

  // ========== EXPORT METHODS ==========
  
  exportConsentData(format: 'json' | 'csv'): void {
    const data = {
      exportDate: this.currentDate.toISOString(),
      forms: this.consentForms(),
      records: this.consentRecords(),
      users: this.userProfiles()
    };
    
    if (format === 'json') {
      this.exportAsJson(data);
    } else {
      this.exportAsCsv(data);
    }
  }
  
  private exportAsJson(data: any): void {
    const dataStr = JSON.stringify(data, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `consent-data-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }
  
  private exportAsCsv(data: any): void {
    // Simplified CSV export for records
    const headers = ['Timestamp', 'User Email', 'Form ID', 'Action', 'Method'];
    const csvRows = data.records.map((record: ConsentRecord) => [
      record.timestamp.toISOString(),
      record.userEmail,
      record.formId,
      record.action,
      record.consentMethod
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvRows.map((row: any[]) => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `consent-records-${this.currentDate.toISOString().slice(0, 10)}.csv`;
    link.click();
  }

  // ========== COMPLIANCE METHODS ==========
  
  runComplianceCheck(): void {
    this.isLoading.set(true);
    
    // Simulate compliance check
    setTimeout(() => {
      const forms = this.consentForms();
      const issues: string[] = [];
      
      forms.forEach(form => {
        if (form.status === 'active') {
          if (form.sections.length === 0) {
            issues.push(`${form.title} has no consent sections`);
          }
          
          if (form.required) {
            const hasExplicitConsent = form.sections.some(s => s.consentType === 'explicit');
            if (!hasExplicitConsent) {
              issues.push(`${form.title} requires explicit consent but doesn't have it`);
            }
          }
          
          form.sections.forEach(section => {
            section.options.forEach(option => {
              if (!option.retentionPeriod || option.retentionPeriod.trim() === '') {
                issues.push(`${form.title} - ${option.label} has no retention period specified`);
              }
            });
          });
        }
      });
      
      if (issues.length > 0) {
        alert(`Compliance check found ${issues.length} issues:\n\n${issues.slice(0, 5).join('\n')}${issues.length > 5 ? '\n...and more' : ''}`);
      } else {
        alert('Compliance check passed! All forms meet requirements.');
      }
      
      this.isLoading.set(false);
    }, 1500);
  }

  // ========== FILTER METHODS ==========
  
  setRegulationFilter(regulation: string): void {
    this.filter.update(f => ({ ...f, regulation }));
  }
  
  setStatusFilter(status: string): void {
    this.filter.update(f => ({ ...f, status }));
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
      regulation: 'all',
      status: 'active',
      dateRange: {
        start: new Date(new Date().setMonth(new Date().getMonth() - 1)),
        end: new Date()
      },
      searchQuery: '',
      consentType: 'all'
    });
  }

  // ========== TEST DATA GENERATION ==========
  
  generateTestConsent(): void {
    const users = this.userProfiles();
    const activeForms = this.consentForms().filter(f => f.status === 'active');
    
    if (users.length === 0 || activeForms.length === 0) {
      console.warn('No users or active forms available for test consent');
      return;
    }
    
    const randomUser = users[Math.floor(Math.random() * users.length)];
    const randomForm = activeForms[Math.floor(Math.random() * activeForms.length)];
    
    // Generate random preferences
    const preferences: ConsentPreference[] = [];
    randomForm.sections.forEach(section => {
      section.options.forEach(option => {
        preferences.push({
          sectionId: section.id,
          optionId: option.id,
          granted: Math.random() > 0.3,
          timestamp: new Date()
        });
      });
    });
    
    this.simulateConsentGrant(randomUser.userId, randomForm.id, preferences);
    
    console.log(`Test consent generated for ${randomUser.email} on ${randomForm.title}`);
  }
}