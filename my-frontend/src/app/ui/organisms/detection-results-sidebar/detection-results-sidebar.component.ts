import { Component, inject, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';
import { Detection, PrivacyFilter } from '../../../shared/interfaces/canvas.interface';

@Component({
  selector: 'app-detection-results-sidebar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './detection-results-sidebar.component.html',
  styleUrls: ['./detection-results-sidebar.component.scss']
})
export class DetectionResultsSidebarComponent {
  // Inject canvas state service
  canvasState = inject(CanvasStateService);
  
  // Define valid detection types for iteration
  detectionTypeKeys: Array<Detection['type']> = ['face', 'text', 'license_plate', 'person', 'vehicle', 'signature'];
  
  // Local state signals
  showOnlySelected = signal(false);
  filterType = signal<'all' | Detection['type']>('all');
  sortBy = signal<'confidence' | 'type' | 'size'>('confidence');
  showConfidenceScores = signal(true);
  
  // Computed signals
  filteredDetections = computed(() => {
    const detections = this.canvasState.detections();
    const filter = this.filterType();
    const onlySelected = this.showOnlySelected();
    const selectedIds = this.canvasState.selectedDetectionIds();
    
    let filtered = detections;
    
    // Filter by type
    if (filter !== 'all') {
      filtered = filtered.filter(d => d.type === filter);
    }
    
    // Filter by selection if enabled
    if (onlySelected) {
      filtered = filtered.filter(d => selectedIds.includes(d.id));
    }
    
    return filtered;
  });
  
  sortedDetections = computed(() => {
    const filtered = this.filteredDetections();
    const sort = this.sortBy();
    
    return [...filtered].sort((a, b) => {
      switch (sort) {
        case 'confidence':
          return b.confidence - a.confidence; // Descending
        case 'type':
          return a.type.localeCompare(b.type);
        case 'size':
          const areaA = a.bbox.width * a.bbox.height;
          const areaB = b.bbox.width * b.bbox.height;
          return areaB - areaA; // Descending
        default:
          return 0;
      }
    });
  });
  
  // Statistics - with index signature to allow string indexing
  detectionStats = computed(() => {
    const detections = this.canvasState.detections();
    const stats: Record<string, number> = {
      'face': 0,
      'text': 0,
      'license_plate': 0,
      'person': 0,
      'vehicle': 0,
      'signature': 0
    };
    
    detections.forEach(d => {
      stats[d.type]++;
    });
    
    return stats;
  });
  
  totalDetections = computed(() => this.canvasState.detections().length);
  selectedCount = computed(() => this.canvasState.selectedDetectionIds().length);
  hasSelected = computed(() => this.selectedCount() > 0);
  
  // Helper methods
  getTypeColor(type: string): string {
    const colors: Record<string, string> = {
      'face': '#3498db',
      'text': '#2ecc71',
      'license_plate': '#e74c3c',
      'person': '#9b59b6',
      'vehicle': '#f39c12',
      'signature': '#1abc9c'
    };
    return colors[type] || '#95a5a6';
  }
  
  getTypeIcon(type: string): string {
    const icons: Record<string, string> = {
      'face': '👤',
      'text': '📝',
      'license_plate': '🚗',
      'person': '🧍',
      'vehicle': '🚙',
      'signature': '✍️'
    };
    return icons[type] || '❓';
  }
  
  // Title case formatter (to replace missing pipe)
  toTitleCase(str: string): string {
    return str.replace(/_/g, ' ')
             .replace(/-/g, ' ')
             .toLowerCase()
             .split(' ')
             .map(word => word.charAt(0).toUpperCase() + word.slice(1))
             .join(' ');
  }
  
  // Selection methods
  toggleDetection(id: string): void {
    this.canvasState.toggleDetectionSelection(id);
  }
  
  selectAll(): void {
    const allIds = this.canvasState.detections().map(d => d.id);
    this.canvasState.setSelectedDetectionIds(allIds);
  }
  
  selectNone(): void {
    this.canvasState.setSelectedDetectionIds([]);
  }
  
  // Bulk filter application
  applyBulkFilter(filter: PrivacyFilter): void {
    this.canvasState.applyFilterToSelected(filter);
  }
  
  // Clear all filters from selected
  clearBulkFilters(): void {
    const selectedIds = this.canvasState.selectedDetectionIds();
    this.canvasState.detections.update(detections =>
      detections.map(detection =>
        selectedIds.includes(detection.id)
          ? { ...detection, appliedFilter: 'none' }
          : detection
      )
    );
  }
  
  // Navigate to detection (center view on it)
  navigateToDetection(detection: Detection): void {
    const bbox = detection.bbox;
    // Simple center calculation - in real app would use canvas transform
    console.log('Navigating to detection:', detection.id, bbox);
    
    // Highlight the detection
    this.canvasState.setSelectedDetectionIds([detection.id]);
  }
  
  // Export detections as JSON
  exportDetections(): void {
    const detections = this.canvasState.detections();
    const dataStr = JSON.stringify(detections, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `detections-${new Date().toISOString().slice(0, 10)}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }
}