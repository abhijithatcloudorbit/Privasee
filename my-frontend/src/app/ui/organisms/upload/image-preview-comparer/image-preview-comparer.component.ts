import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-preview-comparer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-preview-comparer.component.html',
  styleUrls: ['./image-preview-comparer.component.scss']
})
export class ImagePreviewComparerComponent {
  @Input() originalImage: any = null;  // ← ADD THIS
  @Input() processedImage: any = null; // ← ADD THIS
  @Output() manualOverride = new EventEmitter<any>(); // ← ADD THIS
}