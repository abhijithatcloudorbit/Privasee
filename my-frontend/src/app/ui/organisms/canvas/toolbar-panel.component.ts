import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';
import { ToolType, PrivacyFilter } from '../../../shared/interfaces/canvas.interface';

@Component({
  selector: 'app-toolbar-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toolbar-panel.component.html',
  styleUrls: ['./toolbar-panel.component.scss']
})
export class ToolbarPanelComponent {
  private canvasState = inject(CanvasStateService);
  
  // Tool types available
  tools: { id: ToolType; label: string; icon: string }[] = [
    { id: 'select', label: 'Select', icon: '🖱' },
    { id: 'brush', label: 'Brush', icon: '🖌' },
    { id: 'eraser', label: 'Eraser', icon: '🗑' },
    { id: 'pan', label: 'Pan', icon: '✋' },
    { id: 'zoom', label: 'Zoom', icon: '🔍' }
  ];
  
  // Privacy filters available
  filters: { id: PrivacyFilter; label: string }[] = [
    { id: 'blur', label: 'Blur' },
    { id: 'pixelate', label: 'Pixelate' },
    { id: 'redact', label: 'Redact' }
  ];
  
  // Get current selected tool
  get selectedTool() {
    return this.canvasState.selectedTool;
  }
  
  // Check if tool is active
  isToolActive(tool: ToolType): boolean {
    return this.canvasState.selectedTool() === tool;
  }
  
  // Set active tool
  setTool(tool: ToolType): void {
    this.canvasState.setTool(tool);
  }
  
  // Apply filter to selected detections
  applyFilterToSelected(filter: PrivacyFilter): void {
    this.canvasState.applyFilterToSelected(filter);
  }
  
  // Clear all filters
  clearAllFilters(): void {
    this.canvasState.clearAllFilters();
  }
  
  // Reset canvas (clear selections and detections)
  resetCanvas(): void {
    this.canvasState.setSelectedDetectionIds([]);
    this.canvasState.setTool('select');
  }
  
  // Check if any detections are selected
  hasSelectedDetections(): boolean {
    return this.canvasState.selectedDetectionIds().length > 0;
  }
}