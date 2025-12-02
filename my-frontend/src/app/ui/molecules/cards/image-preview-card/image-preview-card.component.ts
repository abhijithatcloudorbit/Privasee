import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../../atoms/buttons/btn-primary.component';
import { BtnSecondaryComponent } from '../../../atoms/buttons/btn-secondary/btn-secondary';
import { BadgeComponent } from '../../../atoms/misc/badge/badge.component';
import { ProgressBarComponent } from '../../../atoms/feedback/progress-bar/progress-bar.component';
import { TextBodyComponent } from '../../../atoms/typography/text-body/text-body.component';

@Component({
  selector: 'app-image-preview-card',
  standalone: true,
  imports: [CommonModule, BtnPrimaryComponent, BtnSecondaryComponent, BadgeComponent, ProgressBarComponent, TextBodyComponent],
  templateUrl: './image-preview-card.component.html',
  styleUrls: ['./image-preview-card.component.scss']
})
export class ImagePreviewCardComponent {
  @Input() imageUrl = '';
  @Input() fileName = '';
  @Input() fileSize = 0;
  @Input() dimensions = '';
  @Input() status: 'pending' | 'processing' | 'completed' | 'error' = 'pending';
  @Input() progress = 0;
  @Input() detectedItems: {type: string, count: number}[] = [];
  
  @Output() preview = new EventEmitter<void>();
  @Output() process = new EventEmitter<void>();
  @Output() remove = new EventEmitter<void>();
  
  getStatusColor(): 'error' | 'primary' | 'success' | 'warning' | 'gray' {
  // Return one of the literal types, not a string
  if (this.status === 'error') return 'error';
  if (this.status === 'completed') return 'success';
  // ... etc
  return 'gray';
}
  
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}