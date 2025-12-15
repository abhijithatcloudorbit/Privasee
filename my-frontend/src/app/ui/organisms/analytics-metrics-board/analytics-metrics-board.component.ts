import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface AccuracyMetric {
  id: 'A1' | 'A2' | 'A3' | 'A4' | 'A5' | 'A6';
  title: string;
  description: string;
  currentValue: number;
  targetValue: number;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  confidenceInterval: [number, number];
  lastUpdated: Date;
  unit: string;
  improvementNeeded: boolean;
  critical: boolean;
}

export interface PerformanceBenchmark {
  name: string;
  value: number;
  ourScore: number;
  better: boolean;
}

@Component({
  selector: 'app-analytics-metrics-board',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './analytics-metrics-board.component.html',
  styleUrls: ['./analytics-metrics-board.component.scss']
})
export class AnalyticsMetricsBoardComponent {
  // Current date for the component
  currentDate = new Date();
  
  // A1-A6 Accuracy Metrics
  accuracyMetrics = signal<AccuracyMetric[]>([
    {
      id: 'A1',
      title: 'Face Detection Accuracy',
      description: 'Precision in detecting human faces across various conditions',
      currentValue: 94.5,
      targetValue: 96.0,
      trend: 'up',
      trendValue: 2.3,
      confidenceInterval: [92.8, 95.7],
      lastUpdated: new Date('2024-01-25'),
      unit: '%',
      improvementNeeded: false,
      critical: true
    },
    {
      id: 'A2',
      title: 'Text Detection Accuracy',
      description: 'Accuracy in detecting and OCR of text elements',
      currentValue: 88.2,
      targetValue: 90.0,
      trend: 'stable',
      trendValue: 0.5,
      confidenceInterval: [86.5, 89.8],
      lastUpdated: new Date('2024-01-24'),
      unit: '%',
      improvementNeeded: true,
      critical: true
    },
    {
      id: 'A3',
      title: 'License Plate Accuracy',
      description: 'Detection and recognition of vehicle license plates',
      currentValue: 96.8,
      targetValue: 95.0,
      trend: 'up',
      trendValue: 3.1,
      confidenceInterval: [95.2, 97.9],
      lastUpdated: new Date('2024-01-26'),
      unit: '%',
      improvementNeeded: false,
      critical: false
    },
    {
      id: 'A4',
      title: 'Person Detection Accuracy',
      description: 'Full body person detection (not just faces)',
      currentValue: 91.3,
      targetValue: 92.0,
      trend: 'up',
      trendValue: 1.8,
      confidenceInterval: [89.7, 92.4],
      lastUpdated: new Date('2024-01-23'),
      unit: '%',
      improvementNeeded: false,
      critical: false
    },
    {
      id: 'A5',
      title: 'Vehicle Detection Accuracy',
      description: 'Vehicle type and make/model identification',
      currentValue: 89.7,
      targetValue: 91.0,
      trend: 'down',
      trendValue: -1.2,
      confidenceInterval: [87.9, 91.1],
      lastUpdated: new Date('2024-01-22'),
      unit: '%',
      improvementNeeded: true,
      critical: false
    },
    {
      id: 'A6',
      title: 'Signature Detection Accuracy',
      description: 'Detection of handwritten signatures in documents',
      currentValue: 82.5,
      targetValue: 85.0,
      trend: 'up',
      trendValue: 4.7,
      confidenceInterval: [80.1, 84.3],
      lastUpdated: new Date('2024-01-21'),
      unit: '%',
      improvementNeeded: true,
      critical: true
    }
  ]);

  // Performance Benchmarks
  performanceBenchmarks = signal<PerformanceBenchmark[]>([
    {
      name: 'Industry Average',
      value: 87.5,
      ourScore: 0,
      better: false
    },
    {
      name: 'Google Vision API',
      value: 92.1,
      ourScore: 0,
      better: false
    },
    {
      name: 'AWS Rekognition',
      value: 90.8,
      ourScore: 0,
      better: false
    },
    {
      name: 'Azure Computer Vision',
      value: 89.7,
      ourScore: 0,
      better: false
    },
    {
      name: 'OpenAI CLIP',
      value: 85.3,
      ourScore: 0,
      better: false
    }
  ]);

  // Computed overall score (average of all metrics)
  overallScore = computed(() => {
    const metrics = this.accuracyMetrics();
    const sum = metrics.reduce((acc, metric) => acc + metric.currentValue, 0);
    return Math.round((sum / metrics.length) * 10) / 10;
  });

  // Computed score for each benchmark
  computedBenchmarks = computed(() => {
    const overall = this.overallScore();
    return this.performanceBenchmarks().map(benchmark => ({
      ...benchmark,
      ourScore: overall,
      better: overall > benchmark.value
    }));
  });

  // Filter signals
  filterCritical = signal(false);
  filterNeedsImprovement = signal(false);
  sortBy = signal<'id' | 'value' | 'trend'>('id');

  // Filtered and sorted metrics
  filteredMetrics = computed(() => {
    let metrics = this.accuracyMetrics();
    
    // Apply filters
    if (this.filterCritical()) {
      metrics = metrics.filter(m => m.critical);
    }
    
    if (this.filterNeedsImprovement()) {
      metrics = metrics.filter(m => m.improvementNeeded);
    }
    
    // Apply sorting
    return [...metrics].sort((a, b) => {
      switch (this.sortBy()) {
        case 'value':
          return b.currentValue - a.currentValue;
        case 'trend':
          const trendOrder = { 'up': 2, 'stable': 1, 'down': 0 };
          return trendOrder[b.trend] - trendOrder[a.trend];
        case 'id':
        default:
          return a.id.localeCompare(b.id);
      }
    });
  });

  // Computed summary stats (FIXED: Move filter logic here from template)
  metricsAbove90 = computed(() => 
    this.accuracyMetrics().filter(m => m.currentValue >= 90).length
  );

  improvingMetrics = computed(() => 
    this.accuracyMetrics().filter(m => m.trend === 'up').length
  );

  needsImprovementMetrics = computed(() => 
    this.accuracyMetrics().filter(m => m.improvementNeeded).length
  );

  criticalMetrics = computed(() => 
    this.accuracyMetrics().filter(m => m.critical).length
  );

  // Helper methods
  getMetricColor(value: number): string {
    if (value >= 95) return '#2ecc71'; // Green
    if (value >= 90) return '#f39c12'; // Orange
    if (value >= 85) return '#e67e22'; // Dark Orange
    return '#e74c3c'; // Red
  }

  getTrendIcon(trend: 'up' | 'down' | 'stable'): string {
    const icons = {
      'up': '📈',
      'down': '📉',
      'stable': '➡️'
    };
    return icons[trend];
  }

  getTrendColor(trend: 'up' | 'down' | 'stable'): string {
    const colors = {
      'up': '#2ecc71',
      'down': '#e74c3c',
      'stable': '#3498db'
    };
    return colors[trend];
  }

  formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  }

  calculateProgress(current: number, target: number): number {
    return Math.min(100, (current / target) * 100);
  }

  getProgressColor(current: number, target: number): string {
    const percentage = (current / target) * 100;
    if (percentage >= 100) return '#2ecc71';
    if (percentage >= 90) return '#f39c12';
    return '#e74c3c';
  }

  // Helper method for rounding (FIXED: Replace direct Math usage in template)
  round(value: number): number {
    return Math.round(value);
  }

  // Export metrics report
  exportMetricsReport(): void {
    const report = {
      generatedAt: this.currentDate.toISOString(),
      overallScore: this.overallScore(),
      metrics: this.accuracyMetrics(),
      benchmarks: this.computedBenchmarks()
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `accuracy-metrics-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }

  // Refresh metrics (simulated)
  refreshMetrics(): void {
    console.log('Refreshing metrics...');
    // In a real app, this would fetch new data
  }

  // Set target for a metric
  setTarget(metricId: string, target: number): void {
    this.accuracyMetrics.update(metrics =>
      metrics.map(metric =>
        metric.id === metricId
          ? { ...metric, targetValue: target }
          : metric
      )
    );
  }
}