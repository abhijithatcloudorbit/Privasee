import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

// Interfaces defined in the component file
interface AccuracyMetric {
  id: string;
  category: 'face' | 'text' | 'license-plate' | 'person' | 'vehicle' | 'signature';
  name: string;
  description: string;
  currentAccuracy: number;
  targetAccuracy: number;
  trend: 'improving' | 'declining' | 'stable';
  trendValue: number; // percentage change
  confidenceInterval: [number, number];
  lastUpdated: Date;
  dataPoints: DataPoint[];
  status: 'excellent' | 'good' | 'warning' | 'critical';
  benchmark: number; // industry benchmark
}

interface DataPoint {
  date: Date;
  accuracy: number;
  volume: number; // number of images processed
  falsePositives: number;
  falseNegatives: number;
}

interface TimePeriod {
  label: string;
  value: '1d' | '7d' | '30d' | '90d' | 'all';
  dataPoints: number;
}

interface Alert {
  id: string;
  type: 'warning' | 'error' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  metricId: string;
  resolved: boolean;
}

@Component({
  selector: 'app-accuracy-dashboard-organism',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './accuracy-dashboard-organism.component.html',
  styleUrls: ['./accuracy-dashboard-organism.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AccuracyDashboardOrganism {
  // Current date for the component
  currentDate = new Date();
  
  // Accuracy Metrics Data
  accuracyMetrics = signal<AccuracyMetric[]>([
    {
      id: 'face-accuracy',
      category: 'face',
      name: 'Face Detection Accuracy',
      description: 'Accuracy in detecting human faces across all conditions',
      currentAccuracy: 94.5,
      targetAccuracy: 96.0,
      trend: 'improving',
      trendValue: 2.3,
      confidenceInterval: [92.8, 95.7],
      lastUpdated: new Date('2024-01-25T10:30:00'),
      dataPoints: this.generateFaceAccuracyData(),
      status: 'good',
      benchmark: 92.1
    },
    {
      id: 'text-accuracy',
      category: 'text',
      name: 'Text Detection Accuracy',
      description: 'Accuracy in detecting and OCR of text elements',
      currentAccuracy: 88.2,
      targetAccuracy: 90.0,
      trend: 'declining',
      trendValue: -1.2,
      confidenceInterval: [86.5, 89.8],
      lastUpdated: new Date('2024-01-24T14:20:00'),
      dataPoints: this.generateTextAccuracyData(),
      status: 'warning',
      benchmark: 87.5
    },
    {
      id: 'license-plate-accuracy',
      category: 'license-plate',
      name: 'License Plate Accuracy',
      description: 'Detection and recognition of vehicle license plates',
      currentAccuracy: 96.8,
      targetAccuracy: 95.0,
      trend: 'improving',
      trendValue: 3.1,
      confidenceInterval: [95.2, 97.9],
      lastUpdated: new Date('2024-01-26T09:15:00'),
      dataPoints: this.generateLicensePlateData(),
      status: 'excellent',
      benchmark: 94.2
    },
    {
      id: 'person-accuracy',
      category: 'person',
      name: 'Person Detection Accuracy',
      description: 'Full body person detection (not just faces)',
      currentAccuracy: 91.3,
      targetAccuracy: 92.0,
      trend: 'improving',
      trendValue: 1.8,
      confidenceInterval: [89.7, 92.4],
      lastUpdated: new Date('2024-01-23T16:45:00'),
      dataPoints: this.generatePersonAccuracyData(),
      status: 'good',
      benchmark: 89.8
    },
    {
      id: 'vehicle-accuracy',
      category: 'vehicle',
      name: 'Vehicle Detection Accuracy',
      description: 'Vehicle type and make/model identification',
      currentAccuracy: 89.7,
      targetAccuracy: 91.0,
      trend: 'declining',
      trendValue: -2.4,
      confidenceInterval: [87.9, 91.1],
      lastUpdated: new Date('2024-01-22T11:30:00'),
      dataPoints: this.generateVehicleAccuracyData(),
      status: 'warning',
      benchmark: 88.5
    },
    {
      id: 'signature-accuracy',
      category: 'signature',
      name: 'Signature Detection Accuracy',
      description: 'Detection of handwritten signatures in documents',
      currentAccuracy: 82.5,
      targetAccuracy: 85.0,
      trend: 'improving',
      trendValue: 4.7,
      confidenceInterval: [80.1, 84.3],
      lastUpdated: new Date('2024-01-21T13:20:00'),
      dataPoints: this.generateSignatureAccuracyData(),
      status: 'critical',
      benchmark: 81.0
    }
  ]);

  // Time Periods
  timePeriods = signal<TimePeriod[]>([
    { label: '24 Hours', value: '1d', dataPoints: 24 },
    { label: '7 Days', value: '7d', dataPoints: 7 },
    { label: '30 Days', value: '30d', dataPoints: 30 },
    { label: '90 Days', value: '90d', dataPoints: 90 },
    { label: 'All Time', value: 'all', dataPoints: 365 }
  ]);

  // Alerts
  alerts = signal<Alert[]>([
    {
      id: 'alert-1',
      type: 'warning',
      title: 'Text Detection Performance Drop',
      message: 'Accuracy dropped by 1.2% in the last 24 hours',
      timestamp: new Date('2024-01-24T14:20:00'),
      metricId: 'text-accuracy',
      resolved: false
    },
    {
      id: 'alert-2',
      type: 'error',
      title: 'Signature Detection Below Target',
      message: 'Current accuracy (82.5%) is below target (85.0%)',
      timestamp: new Date('2024-01-21T13:20:00'),
      metricId: 'signature-accuracy',
      resolved: false
    },
    {
      id: 'alert-3',
      type: 'info',
      title: 'License Plate Detection Improved',
      message: 'Accuracy improved by 3.1% reaching 96.8%',
      timestamp: new Date('2024-01-26T09:15:00'),
      metricId: 'license-plate-accuracy',
      resolved: true
    },
    {
      id: 'alert-4',
      type: 'warning',
      title: 'Vehicle Detection Decline',
      message: 'Accuracy declined by 2.4% in the last week',
      timestamp: new Date('2024-01-22T11:30:00'),
      metricId: 'vehicle-accuracy',
      resolved: false
    }
  ]);

  // UI State Signals
  selectedTimePeriod = signal<'1d' | '7d' | '30d' | '90d' | 'all'>('7d');
  selectedCategory = signal<'all' | 'face' | 'text' | 'license-plate' | 'person' | 'vehicle' | 'signature'>('all');
  sortBy = signal<'accuracy' | 'trend' | 'category' | 'target-gap'>('accuracy');
  showOnlyCritical = signal(false);
  viewMode = signal<'cards' | 'charts' | 'table'>('cards');
  isLoading = signal(false);

  // Computed: Overall accuracy (weighted average)
  overallAccuracy = computed(() => {
    const metrics = this.accuracyMetrics();
    const totalWeight = metrics.reduce((sum, metric) => sum + this.getMetricWeight(metric.category), 0);
    const weightedSum = metrics.reduce((sum, metric) => {
      return sum + (metric.currentAccuracy * this.getMetricWeight(metric.category));
    }, 0);
    
    return Math.round((weightedSum / totalWeight) * 100) / 100;
  });

  // Computed: Target achievement percentage
  targetAchievement = computed(() => {
    const metrics = this.accuracyMetrics();
    const totalTargets = metrics.reduce((sum, metric) => sum + metric.targetAccuracy, 0);
    const totalCurrent = metrics.reduce((sum, metric) => sum + metric.currentAccuracy, 0);
    
    return Math.round((totalCurrent / totalTargets) * 10000) / 100;
  });

  // Computed: Filtered metrics
  filteredMetrics = computed(() => {
    let metrics = this.accuracyMetrics();
    
    // Filter by category
    if (this.selectedCategory() !== 'all') {
      metrics = metrics.filter(metric => metric.category === this.selectedCategory());
    }
    
    // Filter by critical status
    if (this.showOnlyCritical()) {
      metrics = metrics.filter(metric => metric.status === 'critical' || metric.status === 'warning');
    }
    
    // Apply sorting
    return [...metrics].sort((a, b) => {
      switch (this.sortBy()) {
        case 'accuracy':
          return b.currentAccuracy - a.currentAccuracy;
        case 'trend':
          const trendOrder = { 'improving': 2, 'stable': 1, 'declining': 0 };
          return trendOrder[b.trend] - trendOrder[a.trend];
        case 'target-gap':
          const gapA = a.targetAccuracy - a.currentAccuracy;
          const gapB = b.targetAccuracy - b.currentAccuracy;
          return gapB - gapA;
        case 'category':
        default:
          return a.category.localeCompare(b.category);
      }
    });
  });

  // Computed: Summary statistics
  summaryStats = computed(() => {
    const metrics = this.accuracyMetrics();
    
    return {
      totalMetrics: metrics.length,
      aboveTarget: metrics.filter(m => m.currentAccuracy >= m.targetAccuracy).length,
      improving: metrics.filter(m => m.trend === 'improving').length,
      declining: metrics.filter(m => m.trend === 'declining').length,
      excellent: metrics.filter(m => m.status === 'excellent').length,
      good: metrics.filter(m => m.status === 'good').length,
      warning: metrics.filter(m => m.status === 'warning').length,
      critical: metrics.filter(m => m.status === 'critical').length
    };
  });

  // Computed: Active alerts
  activeAlerts = computed(() => {
    return this.alerts().filter(alert => !alert.resolved);
  });

  // Computed: Recent data points for selected time period
  recentDataPoints = computed(() => {
    const period = this.selectedTimePeriod();
    const days = period === '1d' ? 1 : period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 365;
    
    const allDataPoints: DataPoint[] = [];
    this.accuracyMetrics().forEach(metric => {
      metric.dataPoints.forEach(point => {
        const pointDate = new Date(point.date);
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - days);
        
        if (pointDate >= cutoffDate) {
          allDataPoints.push({ ...point, date: pointDate });
        }
      });
    });
    
    return allDataPoints.sort((a, b) => b.date.getTime() - a.date.getTime());
  });

  // Computed: Accuracy trend data for charts
  accuracyTrendData = computed(() => {
    const period = this.selectedTimePeriod();
    const days = period === '1d' ? 1 : period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 365;
    
    const categories = ['face', 'text', 'license-plate', 'person', 'vehicle', 'signature'];
    const trendData: { [key: string]: { date: Date, accuracy: number }[] } = {};
    
    categories.forEach(category => {
      const metric = this.accuracyMetrics().find(m => m.category === category);
      if (metric) {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - days);
        
        trendData[category] = metric.dataPoints
          .filter(point => new Date(point.date) >= cutoffDate)
          .map(point => ({
            date: new Date(point.date),
            accuracy: point.accuracy
          }))
          .sort((a, b) => a.date.getTime() - b.date.getTime());
      }
    });
    
    return trendData;
  });

  // Helper: Get weight for each metric category
  private getMetricWeight(category: string): number {
    const weights: { [key: string]: number } = {
      'face': 1.2,
      'text': 1.0,
      'license-plate': 1.1,
      'person': 0.9,
      'vehicle': 0.8,
      'signature': 0.7
    };
    
    return weights[category] || 1.0;
  }

  // Data generation methods
  private generateFaceAccuracyData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 92 + Math.random() * 5;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 2) * 10) / 10,
        volume: Math.floor(Math.random() * 1000) + 500,
        falsePositives: Math.floor(Math.random() * 20) + 5,
        falseNegatives: Math.floor(Math.random() * 15) + 3
      });
    }
    
    return data;
  }

  private generateTextAccuracyData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 86 + Math.random() * 4;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 3) * 10) / 10,
        volume: Math.floor(Math.random() * 800) + 300,
        falsePositives: Math.floor(Math.random() * 30) + 10,
        falseNegatives: Math.floor(Math.random() * 25) + 8
      });
    }
    
    return data;
  }

  private generateLicensePlateData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 94 + Math.random() * 4;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 2) * 10) / 10,
        volume: Math.floor(Math.random() * 600) + 200,
        falsePositives: Math.floor(Math.random() * 15) + 3,
        falseNegatives: Math.floor(Math.random() * 10) + 2
      });
    }
    
    return data;
  }

  private generatePersonAccuracyData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 89 + Math.random() * 4;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 2) * 10) / 10,
        volume: Math.floor(Math.random() * 700) + 250,
        falsePositives: Math.floor(Math.random() * 25) + 7,
        falseNegatives: Math.floor(Math.random() * 20) + 5
      });
    }
    
    return data;
  }

  private generateVehicleAccuracyData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 88 + Math.random() * 4;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 3) * 10) / 10,
        volume: Math.floor(Math.random() * 500) + 150,
        falsePositives: Math.floor(Math.random() * 20) + 5,
        falseNegatives: Math.floor(Math.random() * 18) + 4
      });
    }
    
    return data;
  }

  private generateSignatureAccuracyData(): DataPoint[] {
    const data: DataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      const baseAccuracy = 80 + Math.random() * 5;
      data.push({
        date,
        accuracy: Math.round((baseAccuracy + (Math.random() - 0.5) * 4) * 10) / 10,
        volume: Math.floor(Math.random() * 400) + 100,
        falsePositives: Math.floor(Math.random() * 35) + 15,
        falseNegatives: Math.floor(Math.random() * 30) + 12
      });
    }
    
    return data;
  }

  // UI Helper Methods
  formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  formatPercentage(value: number): string {
    return value.toFixed(1) + '%';
  }

  getStatusColor(status: string): string {
    const colors: { [key: string]: string } = {
      'excellent': '#2ecc71',
      'good': '#27ae60',
      'warning': '#f39c12',
      'critical': '#e74c3c'
    };
    return colors[status] || '#95a5a6';
  }

  getTrendIcon(trend: string): string {
    const icons: { [key: string]: string } = {
      'improving': '📈',
      'declining': '📉',
      'stable': '➡️'
    };
    return icons[trend] || '➡️';
  }

  getTrendColor(trend: string): string {
    const colors: { [key: string]: string } = {
      'improving': '#2ecc71',
      'declining': '#e74c3c',
      'stable': '#3498db'
    };
    return colors[trend] || '#95a5a6';
  }

  getAlertColor(type: string): string {
    const colors: { [key: string]: string } = {
      'warning': '#f39c12',
      'error': '#e74c3c',
      'info': '#3498db',
      'success': '#2ecc71'
    };
    return colors[type] || '#95a5a6';
  }

  getAlertIcon(type: string): string {
    const icons: { [key: string]: string } = {
      'warning': '⚠️',
      'error': '🚨',
      'info': 'ℹ️',
      'success': '✅'
    };
    return icons[type] || 'ℹ️';
  }

  getCategoryIcon(category: string): string {
    const icons: { [key: string]: string } = {
      'face': '👤',
      'text': '📝',
      'license-plate': '🚗',
      'person': '🚶',
      'vehicle': '🚙',
      'signature': '✍️'
    };
    return icons[category] || '📊';
  }

  // Calculate progress percentage
  calculateProgress(current: number, target: number): number {
    return Math.min(100, (current / target) * 100);
  }

  // Get progress bar color
  getProgressColor(percentage: number): string {
    if (percentage >= 100) return '#2ecc71';
    if (percentage >= 90) return '#27ae60';
    if (percentage >= 80) return '#f39c12';
    if (percentage >= 70) return '#e67e22';
    return '#e74c3c';
  }

  // Calculate accuracy gap
  calculateAccuracyGap(current: number, target: number): string {
    const gap = target - current;
    return gap > 0 ? `-${gap.toFixed(1)}%` : `+${Math.abs(gap).toFixed(1)}%`;
  }

  // Get category name
  getCategoryName(category: string): string {
    const names: { [key: string]: string } = {
      'face': 'Face Detection',
      'text': 'Text Detection',
      'license-plate': 'License Plate',
      'person': 'Person Detection',
      'vehicle': 'Vehicle Detection',
      'signature': 'Signature Detection'
    };
    return names[category] || category;
  }

  // Get max accuracy from last 10 data points
  getMaxAccuracy(metric: AccuracyMetric): number {
    const recentData = metric.dataPoints.slice(-10);
    if (recentData.length === 0) return 0;
    
    let max = recentData[0].accuracy;
    for (let i = 1; i < recentData.length; i++) {
      if (recentData[i].accuracy > max) {
        max = recentData[i].accuracy;
      }
    }
    return Math.round(max * 10) / 10;
  }

  // Get min accuracy from last 10 data points
  getMinAccuracy(metric: AccuracyMetric): number {
    const recentData = metric.dataPoints.slice(-10);
    if (recentData.length === 0) return 0;
    
    let min = recentData[0].accuracy;
    for (let i = 1; i < recentData.length; i++) {
      if (recentData[i].accuracy < min) {
        min = recentData[i].accuracy;
      }
    }
    return Math.round(min * 10) / 10;
  }

  // Get average accuracy from last 10 data points
  getAverageAccuracy(metric: AccuracyMetric): number {
    const recentData = metric.dataPoints.slice(-10);
    if (recentData.length === 0) return 0;
    
    const sum = recentData.reduce((total, point) => total + point.accuracy, 0);
    const count = Math.min(10, metric.dataPoints.length);
    return Math.round((sum / count) * 10) / 10;
  }

  // Round number (replaces Math.round in template)
  roundNumber(value: number): number {
    return Math.round(value);
  }

  // Action Methods
  refreshData(): void {
    this.isLoading.set(true);
    
    // Simulate API call
    setTimeout(() => {
      // Update metrics with slight variations
      this.accuracyMetrics.update(metrics => 
        metrics.map(metric => ({
          ...metric,
          currentAccuracy: Math.max(70, Math.min(99, metric.currentAccuracy + (Math.random() - 0.5) * 2)),
          lastUpdated: new Date()
        }))
      );
      
      // Add a new alert occasionally
      if (Math.random() > 0.7) {
        const metricIds = ['face-accuracy', 'text-accuracy', 'license-plate-accuracy', 'person-accuracy', 'vehicle-accuracy', 'signature-accuracy'];
        const randomMetricId = metricIds[Math.floor(Math.random() * metricIds.length)];
        const metric = this.accuracyMetrics().find(m => m.id === randomMetricId);
        
        if (metric) {
          const newAlert: Alert = {
            id: `alert-${Date.now()}`,
            type: Math.random() > 0.5 ? 'warning' : 'info',
            title: `${metric.name} Update`,
            message: `Accuracy changed to ${metric.currentAccuracy.toFixed(1)}%`,
            timestamp: new Date(),
            metricId: randomMetricId,
            resolved: false
          };
          
          this.alerts.update(alerts => [newAlert, ...alerts.slice(0, 4)]);
        }
      }
      
      this.isLoading.set(false);
    }, 1500);
  }

  resolveAlert(alertId: string): void {
    this.alerts.update(alerts => 
      alerts.map(alert => 
        alert.id === alertId ? { ...alert, resolved: true } : alert
      )
    );
  }

  exportMetricsReport(): void {
    const report = {
      generatedAt: this.currentDate.toISOString(),
      overallAccuracy: this.overallAccuracy(),
      targetAchievement: this.targetAchievement(),
      summary: this.summaryStats(),
      metrics: this.accuracyMetrics().map(metric => ({
        name: metric.name,
        category: metric.category,
        currentAccuracy: metric.currentAccuracy,
        targetAccuracy: metric.targetAccuracy,
        trend: metric.trend,
        status: metric.status,
        lastUpdated: metric.lastUpdated.toISOString()
      }))
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `accuracy-dashboard-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }

  // Toggle view mode
  setViewMode(mode: 'cards' | 'charts' | 'table'): void {
    this.viewMode.set(mode);
  }

  // Get accuracy data for trend charts
  getChartDataPoints(metricId: string): { x: number, y: number }[] {
    const metric = this.accuracyMetrics().find(m => m.id === metricId);
    if (!metric) return [];
    
    const recentData = metric.dataPoints.slice(-10);
    
    return recentData.map((point, index) => ({
      x: index * 40 + 20,
      y: 120 - (point.accuracy / 100) * 100
    }));
  }

  // Get chart path for SVG
  getChartPath(points: { x: number, y: number }[]): string {
    if (points.length === 0) return '';
    
    return 'M ' + points.map(point => `${point.x},${point.y}`).join(' L ');
  }

  // Get benchmark comparison
  getBenchmarkComparison(current: number, benchmark: number): string {
    const difference = current - benchmark;
    return difference >= 0 ? `+${difference.toFixed(1)}% above` : `${difference.toFixed(1)}% below`;
  }

  // Get benchmark color
  getBenchmarkColor(current: number, benchmark: number): string {
    return current >= benchmark ? '#2ecc71' : '#e74c3c';
  }
}