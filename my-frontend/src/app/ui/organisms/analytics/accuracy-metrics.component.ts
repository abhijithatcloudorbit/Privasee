// ui/organisms/analytics/accuracy-metrics.component.ts
import { Component, inject, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../atoms/buttons/btn-primary.component';
import { BadgeComponent } from '../../atoms/misc/badge/badge.component';
import { ProgressBarComponent } from '../../atoms/feedback/progress-bar/progress-bar.component';
import { DividerComponent } from '../../atoms/misc/divider/divider.component';


@Component({
  selector: 'app-accuracy-metrics',
  standalone: true,
  imports: [
    CommonModule,
    BtnPrimaryComponent,
    BadgeComponent,
    ProgressBarComponent,
    DividerComponent
  ],
  templateUrl: './accuracy-metrics.component.html',
  styleUrls: ['./accuracy-metrics.component.scss']
})
export class AccuracyMetricsComponent {
  // Mock data for now
  accuracyMetrics = signal({
    a1: 95.2, // Face detection accuracy
    a2: 87.6, // License plate accuracy  
    a3: 92.1, // Text detection accuracy
    a4: 98.3  // Signature detection accuracy
  });
  
  selectedMetric = signal<'a1' | 'a2' | 'a3' | 'a4'>('a1');
  timeRange = signal<'day' | 'week' | 'month'>('week');
  
  // Mock historical data
  historicalData = signal({
    a1: [92, 94, 93, 95, 96, 95, 94],
    a2: [85, 86, 87, 88, 87, 88, 87],
    a3: [90, 91, 92, 92, 93, 92, 92],
    a4: [97, 98, 98, 99, 98, 99, 98]
  });
  
  currentAccuracy = computed(() => {
    const metric = this.selectedMetric();
    return this.accuracyMetrics()[metric];
  });
  
  accuracyLabel = computed(() => {
    const labels = {
      a1: 'Face Detection',
      a2: 'License Plate',
      a3: 'Text Recognition',
      a4: 'Signature Detection'
    };
    return labels[this.selectedMetric()];
  });
  
  accuracyVariant = computed(() => {
    const value = this.currentAccuracy();
    if (value >= 95) return 'success';
    if (value >= 85) return 'primary';
    if (value >= 75) return 'warning';
    return 'error';
  });

  selectMetric(metric: 'a1' | 'a2' | 'a3' | 'a4'): void {
    this.selectedMetric.set(metric);
  }

  setTimeRange(range: 'day' | 'week' | 'month'): void {
    this.timeRange.set(range);
  }

  // Mock methods
  getAverageProcessingTime(): number {
    return 142; // ms
  }

  getDetectionCount(): number {
    return 24;
  }
}