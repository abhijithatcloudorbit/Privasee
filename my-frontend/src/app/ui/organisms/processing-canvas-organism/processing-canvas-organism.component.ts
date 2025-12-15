import { Component, input, effect, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';
import { ImageProcessingService } from '../../../services/image-processing.service';
import { Detection, PrivacyFilter } from '../../../shared/interfaces/canvas.interface';
import { BatchProcessingItem } from '../../../shared/interfaces/batch.interface';

@Component({
  selector: 'app-processing-canvas-organism',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './processing-canvas-organism.component.html',
  styleUrls: ['./processing-canvas-organism.component.scss']
})
export class ProcessingCanvasOrganismComponent {
  // Input for batch item
  batchItem = input<BatchProcessingItem | null>(null);
  
  // Inject services
  private canvasState = inject(CanvasStateService);
  private imageProcessing = inject(ImageProcessingService);
  
  // Local state
  currentImage = signal<string | null>(null);
  loading = signal(false);
  
  // Getters for template
  get hasImage() {
    return () => !!this.currentImage();
  }
  
  get detections() {
    return this.canvasState.detections;
  }
  
  get selectedDetectionIds() {
    return this.canvasState.selectedDetectionIds;
  }
  
  constructor() {
    // React to batchItem changes
    effect(() => {
      const item = this.batchItem();
      if (item && item.selectedFile) {
        this.loadImage(item.selectedFile);
      } else {
        this.resetCanvas();
      }
    });
  }
  
  private async loadImage(file: any): Promise<void> {
    this.loading.set(true);
    
    try {
      if (file.url) {
        this.currentImage.set(file.url);
        this.canvasState.image.set(file.url);
        
        // Run AI detection if no detections exist
        if (!file.detections || file.detections.length === 0) {
          const detections = await this.imageProcessing.detectObjects(file.url);
          this.canvasState.setDetections(detections);
        } else {
          this.canvasState.setDetections(file.detections);
        }
      }
    } catch (error) {
      console.error('Failed to load image:', error);
    } finally {
      this.loading.set(false);
    }
  }
  
  private resetCanvas(): void {
    this.currentImage.set(null);
    this.canvasState.image.set(null);
    this.canvasState.clearAllDetections();
    this.canvasState.setSelectedDetectionIds([]);
  }
  
  // Template methods
  isDetectionSelected(detectionId: string): boolean {
    return this.selectedDetectionIds().includes(detectionId);
  }
  
  selectDetection(detectionId: string): void {
    this.canvasState.toggleDetectionSelection(detectionId);
  }
  
  applyFilter(detectionId: string, filter: PrivacyFilter): void {
    // First select the detection
    this.canvasState.setSelectedDetectionIds([detectionId]);
    // Then apply filter
    this.canvasState.applyFilterToSelected(filter);
  }
  
  // Calculate position for overlay
  getDetectionStyle(detection: Detection) {
    return {
      left: detection.bbox.x + 'px',
      top: detection.bbox.y + 'px',
      width: detection.bbox.width + 'px',
      height: detection.bbox.height + 'px'
    };
  }
  
  getDetectionClass(detection: Detection): string {
    const base = 'c-processing-canvas__detection';
    const selected = this.isDetectionSelected(detection.id) ? 'c-processing-canvas__detection--selected' : '';
    const type = `c-processing-canvas__detection--${detection.type}`;
    const filtered = detection.appliedFilter && detection.appliedFilter !== 'none' 
      ? `c-processing-canvas__detection--${detection.appliedFilter}` 
      : '';
    
    return `${base} ${type} ${selected} ${filtered}`.trim();
  }
}