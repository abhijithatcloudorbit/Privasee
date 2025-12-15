import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';
import { ToolType, PrivacyFilter } from '../../../shared/interfaces/canvas.interface';

@Component({
  selector: 'app-manual-override-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './manual-override-panel.component.html',
  styleUrls: ['./manual-override-panel.component.scss']
})
export class ManualOverridePanelComponent {
  // Inject canvas state service
  canvasState = inject(CanvasStateService);
  
  // Tool types
  tools = signal<{id: ToolType; label: string; icon: string}[]>([
    { id: 'select', label: 'Select', icon: '🖱' },
    { id: 'brush', label: 'Brush', icon: '🖌' },
    { id: 'eraser', label: 'Eraser', icon: '🗑' },
    { id: 'pan', label: 'Pan', icon: '✋' },
    { id: 'zoom', label: 'Zoom', icon: '🔍' }
  ]);
  
  // Brush settings
  brushSize = signal(20);
  brushOpacity = signal(80);
  brushHardness = signal(50);
  
  // History tracking (simplified)
  canUndo = signal(false);
  canRedo = signal(false);
  
  // Quick filters
  quickFilters = signal<{id: PrivacyFilter; label: string; icon: string}[]>([
    { id: 'blur', label: 'Blur Area', icon: '🌀' },
    { id: 'pixelate', label: 'Pixelate Area', icon: '🔳' },
    { id: 'redact', label: 'Redact Area', icon: '⬛' }
  ]);
  
  // Current active tool (connected to canvas state)
  get activeTool() {
    return this.canvasState.selectedTool;
  }
  
  // Is the given tool active?
  isToolActive(tool: ToolType): boolean {
    return this.activeTool() === tool;
  }
  
  // Set active tool
  setTool(tool: ToolType): void {
    this.canvasState.setTool(tool);
  }
  
  // Update brush size
  updateBrushSize(size: number): void {
    this.brushSize.set(Math.max(1, Math.min(100, size)));
  }
  
  // Update brush opacity
  updateBrushOpacity(opacity: number): void {
    this.brushOpacity.set(Math.max(0, Math.min(100, opacity)));
  }
  
  // Update brush hardness
  updateBrushHardness(hardness: number): void {
    this.brushHardness.set(Math.max(0, Math.min(100, hardness)));
  }
  
  // Apply quick filter to canvas (for manual area selection)
  applyQuickFilter(filter: PrivacyFilter): void {
    console.log(`Applying ${filter} filter to manually selected area`);
    // This would trigger a manual filter application in the canvas
    // For now, we'll just apply to selected detections
    this.canvasState.applyFilterToSelected(filter);
  }
  
  // Undo last manual edit
  undo(): void {
    console.log('Undo last edit');
    this.canUndo.set(false); // Simplified - would track actual history
  }
  
  // Redo last undone edit
  redo(): void {
    console.log('Redo last undone edit');
    this.canRedo.set(false); // Simplified - would track actual history
  }
  
  // Clear all manual edits
  clearAllEdits(): void {
    console.log('Clearing all manual edits');
    this.canvasState.clearAllFilters();
  }
  
  // Reset brush to defaults
  resetBrush(): void {
    this.brushSize.set(20);
    this.brushOpacity.set(80);
    this.brushHardness.set(50);
  }
  
  // Get brush preview style
  getBrushPreviewStyle(): any {
    const size = this.brushSize();
    const opacity = this.brushOpacity() / 100;
    return {
      width: `${size}px`,
      height: `${size}px`,
      opacity: opacity,
      'border-radius': `${this.brushHardness()}%`
    };
  }
  
  // Format percentage
  formatPercent(value: number): string {
    return `${value}%`;
  }
}