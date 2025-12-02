import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { InputTextComponent } from '../../../atoms/inputs/input-text/input-text.component';
import { ChipComponent } from '../../../atoms/misc/chip/chip.component';
import { BadgeComponent } from '../../../atoms/misc/badge/badge.component';
import { TextBodyComponent } from '../../../atoms/typography/text-body/text-body.component';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-industry-select-form',
  standalone: true,
  imports: [FormsModule, CommonModule, InputTextComponent, ChipComponent, BadgeComponent, TextBodyComponent],
  templateUrl: './industry-select-form.component.html',
  styleUrls: ['./industry-select-form.component.scss']
})
export class IndustrySelectFormComponent {
  @Input() selectedIndustries: string[] = [];
  @Output() industriesChange = new EventEmitter<string[]>();
  
  industries = [
    { id: 'healthcare', label: 'Healthcare', icon: '🏥', description: 'Patient data privacy (HIPAA/GDPR)' },
    { id: 'automotive', label: 'Automotive', icon: '🚗', description: 'Dashcam & sensor data privacy' },
    { id: 'manufacturing', label: 'Manufacturing', icon: '🏭', description: 'IP protection & factory audits' },
    { id: 'insurance', label: 'Insurance', icon: '📄', description: 'Claim verification & PII protection' },
    { id: 'pharma', label: 'Pharmaceutical', icon: '💊', description: 'Clinical trial data anonymization' },
    { id: 'research', label: 'Research', icon: '🔬', description: 'Academic & research data privacy' },
  ];
  
  searchQuery = '';
  
  getIndustry(industryId: string) {
    return this.industries.find(industry => industry.id === industryId);
  }
  
  toggleIndustry(industryId: string): void {
    if (this.selectedIndustries.includes(industryId)) {
      this.selectedIndustries = this.selectedIndustries.filter(id => id !== industryId);
    } else {
      this.selectedIndustries = [...this.selectedIndustries, industryId];
    }
    this.industriesChange.emit(this.selectedIndustries);
  }
  
  get filteredIndustries() {
    return this.industries.filter(industry =>
      industry.label.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
      industry.description.toLowerCase().includes(this.searchQuery.toLowerCase())
    );
  }
}