import { Injectable, signal } from '@angular/core';
import { Detection, ToolType, PrivacyFilter, CanvasTransform } from '../../shared/interfaces/canvas.interface';

@Injectable({
  providedIn: 'root'
})
export class CanvasStateService {
  // Image state
  image = signal<string | null>(null);
  
  // Detections state
  detections = signal<Detection[]>([]);
  selectedDetectionIds = signal<string[]>([]);
  
  // Canvas transform state
  transform = signal<CanvasTransform>({ x: 0, y: 0, scale: 1 });
  zoom = signal<number>(1);
  
  // Tool state
  selectedTool = signal<ToolType>('select');
  
  // Processing state
  isProcessing = signal<boolean>(false);

  // Methods
  setDetections(detections: Detection[]): void {
    this.detections.set(detections);
  }

  addDetection(detection: Detection): void {
    this.detections.update(current => [...current, detection]);
  }

  removeDetection(id: string): void {
    this.detections.update(current => current.filter(d => d.id !== id));
  }

  clearAllDetections(): void {
    this.detections.set([]);
  }

  setSelectedDetectionIds(ids: string[]): void {
    this.selectedDetectionIds.set(ids);
  }

  toggleDetectionSelection(id: string): void {
  this.selectedDetectionIds.update(current => {
    if (current.includes(id)) {
      return current.filter(i => i !== id);
    } else {
      return [...current, id];
    }
    });
  }

  setTool(tool: ToolType): void {
    this.selectedTool.set(tool);
  }

  applyFilterToSelected(filter: PrivacyFilter): void {
    const selectedIds = this.selectedDetectionIds();
    this.detections.update(detections =>
      detections.map(detection => 
        selectedIds.includes(detection.id) 
          ? { ...detection, appliedFilter: filter }
          : detection
      )
    );
  }

  clearAllFilters(): void {
    this.detections.update(detections =>
      detections.map(detection => ({ ...detection, appliedFilter: 'none' }))
    );
  }

  setTransform(transform: CanvasTransform): void {
    this.transform.set(transform);
  }

  setZoom(zoom: number): void {
    this.zoom.set(zoom);
  }

  pan(deltaX: number, deltaY: number): void {
    this.transform.update(current => ({
      ...current,
      x: current.x + deltaX,
      y: current.y + deltaY
    }));
  }

  scale(factor: number, centerX?: number, centerY?: number): void {
    this.transform.update(current => ({
      ...current,
      scale: current.scale * factor
    }));
    this.zoom.update(current => current * factor);
  }

  resetTransform(): void {
    this.transform.set({ x: 0, y: 0, scale: 1 });
    this.zoom.set(1);
  }
}