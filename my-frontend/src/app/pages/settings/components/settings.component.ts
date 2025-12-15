import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../../ui/atoms/buttons/btn-primary.component';
import { BtnSecondaryComponent } from '../../../ui/atoms/buttons/btn-secondary/btn-secondary';
import { CardBasicComponent } from '../../../ui/atoms/card-basic/card-basic';


export interface Setting {
  id: string;
  label: string;
  description: string;
  type: 'toggle' | 'number' | 'select' | 'text';
  value: boolean | number | string;
  options?: { label: string; value: string }[];
  min?: number;
  max?: number;
  step?: number;
  category: 'ai' | 'privacy' | 'performance' | 'export';
}

@Component({
  selector: 'app-settings',
  standalone: true,
  imports: [CommonModule, BtnPrimaryComponent, BtnSecondaryComponent, CardBasicComponent],
  templateUrl: './settings.component.html',
  styleUrls: ['./settings.component.scss']
})
export class SettingsComponent {
  // Signal-based settings
  settings = signal<Setting[]>([
    {
      id: 'confidence_threshold',
      label: 'Confidence Threshold',
      description: 'Minimum confidence score for AI detections',
      type: 'number',
      value: 0.75,
      min: 0.1,
      max: 1.0,
      step: 0.05,
      category: 'ai'
    },
    {
      id: 'auto_detect_faces',
      label: 'Auto-detect Faces',
      description: 'Automatically detect and blur faces',
      type: 'toggle',
      value: true,
      category: 'privacy'
    },
    {
      id: 'auto_detect_text',
      label: 'Auto-detect Text',
      description: 'Automatically detect and redact sensitive text',
      type: 'toggle',
      value: true,
      category: 'privacy'
    },
    {
      id: 'default_filter',
      label: 'Default Privacy Filter',
      description: 'Default filter to apply to detections',
      type: 'select',
      value: 'blur',
      options: [
        { label: 'Blur', value: 'blur' },
        { label: 'Pixelate', value: 'pixelate' },
        { label: 'Redact', value: 'redact' },
        { label: 'Mask', value: 'mask' }
      ],
      category: 'privacy'
    },
    {
      id: 'batch_size',
      label: 'Batch Processing Size',
      description: 'Number of images to process simultaneously',
      type: 'number',
      value: 10,
      min: 1,
      max: 50,
      step: 1,
      category: 'performance'
    },
    {
      id: 'export_format',
      label: 'Export Format',
      description: 'Default format for processed images',
      type: 'select',
      value: 'png',
      options: [
        { label: 'PNG', value: 'png' },
        { label: 'JPEG', value: 'jpeg' },
        { label: 'WebP', value: 'webp' }
      ],
      category: 'export'
    },
    {
      id: 'retain_metadata',
      label: 'Retain Metadata',
      description: 'Keep original image metadata after processing',
      type: 'toggle',
      value: false,
      category: 'privacy'
    },
    {
      id: 'watermark',
      label: 'Add Watermark',
      description: 'Add privacy compliance watermark to processed images',
      type: 'toggle',
      value: true,
      category: 'export'
    }
  ]);

  // Computed signals for filtered settings
  aiSettings = computed(() => this.settings().filter(s => s.category === 'ai'));
  privacySettings = computed(() => this.settings().filter(s => s.category === 'privacy'));
  performanceSettings = computed(() => this.settings().filter(s => s.category === 'performance'));
  exportSettings = computed(() => this.settings().filter(s => s.category === 'export'));

  // Methods for updating settings
  updateSetting(id: string, value: boolean | number | string): void {
    this.settings.update(settings =>
      settings.map(setting =>
        setting.id === id ? { ...setting, value } : setting
      )
    );
  }

  handleToggleChange(id: string, event: Event): void {
    const target = event.target as HTMLInputElement;
    this.updateSetting(id, target.checked);
  }

  handleNumberChange(id: string, event: Event): void {
    const target = event.target as HTMLInputElement;
    const value = parseFloat(target.value);
    if (!isNaN(value)) {
      this.updateSetting(id, value);
    }
  }

  handleSelectChange(id: string, event: Event): void {
    const target = event.target as HTMLSelectElement;
    this.updateSetting(id, target.value);
  }

  resetToDefaults(): void {
    this.settings.set([
      { id: 'confidence_threshold', label: 'Confidence Threshold', description: 'Minimum confidence score for AI detections', type: 'number', value: 0.75, min: 0.1, max: 1.0, step: 0.05, category: 'ai' },
      { id: 'auto_detect_faces', label: 'Auto-detect Faces', description: 'Automatically detect and blur faces', type: 'toggle', value: true, category: 'privacy' },
      { id: 'auto_detect_text', label: 'Auto-detect Text', description: 'Automatically detect and redact sensitive text', type: 'toggle', value: true, category: 'privacy' },
      { id: 'default_filter', label: 'Default Privacy Filter', description: 'Default filter to apply to detections', type: 'select', value: 'blur', options: [{ label: 'Blur', value: 'blur' }, { label: 'Pixelate', value: 'pixelate' }, { label: 'Redact', value: 'redact' }, { label: 'Mask', value: 'mask' }], category: 'privacy' },
      { id: 'batch_size', label: 'Batch Processing Size', description: 'Number of images to process simultaneously', type: 'number', value: 10, min: 1, max: 50, step: 1, category: 'performance' },
      { id: 'export_format', label: 'Export Format', description: 'Default format for processed images', type: 'select', value: 'png', options: [{ label: 'PNG', value: 'png' }, { label: 'JPEG', value: 'jpeg' }, { label: 'WebP', value: 'webp' }], category: 'export' },
      { id: 'retain_metadata', label: 'Retain Metadata', description: 'Keep original image metadata after processing', type: 'toggle', value: false, category: 'privacy' },
      { id: 'watermark', label: 'Add Watermark', description: 'Add privacy compliance watermark to processed images', type: 'toggle', value: true, category: 'export' }
    ]);
  }

  saveSettings(): void {
    console.log('Settings saved:', this.settings());
    alert('Settings saved successfully!');
  }
}