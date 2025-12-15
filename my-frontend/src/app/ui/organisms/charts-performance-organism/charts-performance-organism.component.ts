import { Component, signal, computed, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';

// Interfaces defined directly in the component file
interface PerformanceChart {
  id: string;
  title: string;
  description: string;
  type: 'line' | 'bar' | 'area' | 'scatter';
  data: ChartDataPoint[];
  xAxisLabel: string;
  yAxisLabel: string;
  timeRange: TimeRange;
  p5Value: number;
  p50Value: number;
  p95Value: number;
  color: string;
  unit: string;
  trend: 'up' | 'down' | 'stable';
}

interface ChartDataPoint {
  timestamp: Date;
  value: number;
  category?: string;
  metadata?: {
    batchId?: string;
    imageCount?: number;
    processingTime?: number;
  };
  // For scatter plots
  xValue?: number;
  yValue?: number;
}

type TimeRange = '1h' | '24h' | '7d' | '30d' | 'all';
type ChartType = 'line' | 'bar' | 'area' | 'scatter';
type SortOrder = 'asc' | 'desc';

@Component({
  selector: 'app-charts-performance-organism',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './charts-performance-organism.component.html',
  styleUrls: ['./charts-performance-organism.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChartsPerformanceOrganism {
  // Current date for the component
  currentDate = new Date();
  
  // Time ranges for the UI
  timeRanges = signal<{value: TimeRange, label: string}[]>([
    { value: '1h', label: '1 Hour' },
    { value: '24h', label: '24 Hours' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
    { value: 'all', label: 'All Time' }
  ]);

  // Performance Charts Data
  performanceCharts = signal<PerformanceChart[]>([
    {
      id: 'processing-time',
      title: 'Processing Time Distribution',
      description: 'P5, P50, P95 processing times across batches',
      type: 'bar',
      data: this.generateProcessingTimeData(),
      xAxisLabel: 'Time Period',
      yAxisLabel: 'Processing Time (ms)',
      timeRange: '24h',
      p5Value: 420,
      p50Value: 185,
      p95Value: 85,
      color: '#3498db',
      unit: 'ms',
      trend: 'up'
    },
    {
      id: 'memory-usage',
      title: 'Memory Usage Over Time',
      description: 'RAM consumption during image processing',
      type: 'area',
      data: this.generateMemoryUsageData(),
      xAxisLabel: 'Timestamp',
      yAxisLabel: 'Memory Usage (MB)',
      timeRange: '24h',
      p5Value: 2450,
      p50Value: 1870,
      p95Value: 1250,
      color: '#2ecc71',
      unit: 'MB',
      trend: 'down'
    },
    {
      id: 'detection-accuracy',
      title: 'Detection Accuracy Trends',
      description: 'A1-A6 accuracy metrics over time',
      type: 'line',
      data: this.generateAccuracyData(),
      xAxisLabel: 'Date',
      yAxisLabel: 'Accuracy (%)',
      timeRange: '7d',
      p5Value: 82.5,
      p50Value: 89.3,
      p95Value: 96.8,
      color: '#9b59b6',
      unit: '%',
      trend: 'up'
    },
    {
      id: 'batch-latency',
      title: 'Batch Processing Latency',
      description: 'Latency distribution across batch sizes',
      type: 'scatter',
      data: this.generateLatencyData(),
      xAxisLabel: 'Batch Size',
      yAxisLabel: 'Latency (s)',
      timeRange: '30d',
      p5Value: 12.5,
      p50Value: 8.2,
      p95Value: 4.1,
      color: '#e74c3c',
      unit: 's',
      trend: 'stable'
    },
    {
      id: 'cpu-utilization',
      title: 'CPU Utilization',
      description: 'CPU usage during AI inference',
      type: 'line',
      data: this.generateCPUData(),
      xAxisLabel: 'Time',
      yAxisLabel: 'CPU (%)',
      timeRange: '1h',
      p5Value: 92,
      p50Value: 65,
      p95Value: 42,
      color: '#f39c12',
      unit: '%',
      trend: 'down'
    },
    {
      id: 'gpu-utilization',
      title: 'GPU Utilization',
      description: 'GPU usage during model inference',
      type: 'area',
      data: this.generateGPUData(),
      xAxisLabel: 'Time',
      yAxisLabel: 'GPU (%)',
      timeRange: '1h',
      p5Value: 88,
      p50Value: 72,
      p95Value: 58,
      color: '#1abc9c',
      unit: '%',
      trend: 'stable'
    }
  ]);

  // UI State Signals
  selectedTimeRange = signal<TimeRange>('24h');
  selectedChartType = signal<ChartType | 'all'>('all');
  selectedChartId = signal<string | null>('processing-time');
  sortOrder = signal<SortOrder>('desc');
  zoomLevel = signal<number>(100);
  isFullscreen = signal<boolean>(false);
  isLoading = signal<boolean>(false);

  // Computed: Filter charts by time range and type
  filteredCharts = computed(() => {
    let charts = this.performanceCharts();
    
    // Filter by time range
    if (this.selectedTimeRange() !== 'all') {
      charts = charts.filter(chart => chart.timeRange === this.selectedTimeRange());
    }
    
    // Filter by chart type
    if (this.selectedChartType() !== 'all') {
      charts = charts.filter(chart => chart.type === this.selectedChartType());
    }
    
    return charts;
  });

  // Computed: Get active chart
  activeChart = computed(() => {
    const selectedId = this.selectedChartId();
    if (!selectedId) return this.filteredCharts()[0];
    
    const foundChart = this.filteredCharts().find(chart => chart.id === selectedId);
    return foundChart || this.filteredCharts()[0];
  });

  // Computed: Get visible charts
  visibleCharts = computed(() => {
    return this.filteredCharts();
  });

  // Computed: Average P5 value across all charts
  averageP5 = computed(() => {
    const charts = this.performanceCharts();
    const sum = charts.reduce((acc, chart) => acc + chart.p5Value, 0);
    return Math.round((sum / charts.length) * 100) / 100;
  });

  // Computed: Performance summary stats
  performanceSummary = computed(() => {
    const charts = this.performanceCharts();
    
    return {
      totalCharts: charts.length,
      improving: charts.filter(c => c.trend === 'up').length,
      stable: charts.filter(c => c.trend === 'stable').length,
      declining: charts.filter(c => c.trend === 'down').length,
      bestP5: Math.min(...charts.map(c => c.p5Value)),
      worstP5: Math.max(...charts.map(c => c.p5Value)),
    };
  });

  // Computed: Chart type counts with proper type
  chartTypeCounts = computed(() => {
    const charts = this.performanceCharts();
    const counts: Record<ChartType, number> = {
      line: 0,
      bar: 0,
      area: 0,
      scatter: 0
    };
    
    charts.forEach(chart => {
      counts[chart.type] = (counts[chart.type] || 0) + 1;
    });
    
    return counts;
  });

  // Computed: Current chart index for navigation
  currentChartIndex = computed(() => {
    const charts = this.filteredCharts();
    const selectedId = this.selectedChartId();
    
    if (!selectedId) return 1;
    
    const index = charts.findIndex(chart => chart.id === selectedId);
    return index >= 0 ? index + 1 : 1;
  });

  // Computed: Total filtered charts count
  totalFilteredCharts = computed(() => {
    return this.filteredCharts().length;
  });

  // Methods for data generation (mock data)
  private generateProcessingTimeData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    
    for (let i = 0; i < 24; i++) {
      const time = new Date(now.getTime() - (23 - i) * 3600000);
      data.push({
        timestamp: time,
        value: Math.floor(Math.random() * 200) + 50,
        metadata: {
          batchId: `BATCH-${1000 + i}`,
          imageCount: Math.floor(Math.random() * 50) + 10
        }
      });
    }
    
    return data;
  }

  private generateMemoryUsageData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    
    for (let i = 0; i < 24; i++) {
      const time = new Date(now.getTime() - (23 - i) * 3600000);
      data.push({
        timestamp: time,
        value: Math.floor(Math.random() * 1500) + 1000,
        metadata: {
          processingTime: Math.floor(Math.random() * 300) + 100
        }
      });
    }
    
    return data;
  }

  private generateAccuracyData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    
    for (let i = 0; i < 7; i++) {
      const time = new Date(now.getTime() - (6 - i) * 86400000);
      data.push({
        timestamp: time,
        value: 85 + Math.random() * 15,
        category: ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'][i % 6]
      });
    }
    
    return data;
  }

  private generateLatencyData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    
    for (let i = 0; i < 20; i++) {
      const batchSize = Math.floor(Math.random() * 100) + 10;
      data.push({
        timestamp: new Date(),
        value: 5 + (batchSize * 0.1) + Math.random() * 3,
        xValue: batchSize, // For scatter plot
        yValue: 5 + (batchSize * 0.1) + Math.random() * 3, // For scatter plot
        metadata: {
          batchId: `BATCH-${2000 + i}`,
          imageCount: batchSize
        }
      });
    }
    
    return data;
  }

  private generateCPUData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    
    for (let i = 0; i < 12; i++) {
      const time = new Date(now.getTime() - (11 - i) * 300000);
      data.push({
        timestamp: time,
        value: Math.floor(Math.random() * 40) + 50,
      });
    }
    
    return data;
  }

  private generateGPUData(): ChartDataPoint[] {
    const data: ChartDataPoint[] = [];
    const now = new Date();
    
    for (let i = 0; i < 12; i++) {
      const time = new Date(now.getTime() - (11 - i) * 300000);
      data.push({
        timestamp: time,
        value: Math.floor(Math.random() * 30) + 60,
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

  formatNumber(value: number): string {
    if (value >= 1000) {
      return (value / 1000).toFixed(1) + 'k';
    }
    return value.toFixed(1);
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

  // Chart Interaction Methods
  selectChart(chartId: string): void {
    this.selectedChartId.set(chartId);
  }

  changeTimeRange(range: string): void {
    // Type guard to ensure it's a valid TimeRange
    if (['1h', '24h', '7d', '30d', 'all'].includes(range)) {
      this.selectedTimeRange.set(range as TimeRange);
    }
  }

  changeChartType(type: string): void {
    // Type guard to ensure it's a valid ChartType or 'all'
    if (['line', 'bar', 'area', 'scatter', 'all'].includes(type)) {
      this.selectedChartType.set(type as ChartType | 'all');
    }
  }

  toggleSortOrder(): void {
    this.sortOrder.set(this.sortOrder() === 'asc' ? 'desc' : 'asc');
  }

  zoomIn(): void {
    this.zoomLevel.update(level => Math.min(level + 25, 200));
  }

  zoomOut(): void {
    this.zoomLevel.update(level => Math.max(level - 25, 50));
  }

  resetZoom(): void {
    this.zoomLevel.set(100);
  }

  toggleFullscreen(): void {
    this.isFullscreen.update(state => !state);
  }

  // Data Export Methods
  exportChartData(chartId: string): void {
    const chart = this.performanceCharts().find(c => c.id === chartId);
    if (!chart) return;

    const csvData = chart.data.map(point => ({
      timestamp: point.timestamp.toISOString(),
      value: point.value,
      category: point.category || '',
      batchId: point.metadata?.batchId || '',
      imageCount: point.metadata?.imageCount || '',
      processingTime: point.metadata?.processingTime || ''
    }));

    const csvContent = [
      Object.keys(csvData[0]).join(','),
      ...csvData.map(row => Object.values(row).map(v => v ?? '').join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${chart.id}-data-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
  }

  exportAllCharts(): void {
    const allData = {
      exportDate: this.currentDate.toISOString(),
      charts: this.performanceCharts().map(chart => ({
        id: chart.id,
        title: chart.title,
        p5Value: chart.p5Value,
        p50Value: chart.p50Value,
        p95Value: chart.p95Value,
        dataPoints: chart.data.length
      }))
    };

    const dataStr = JSON.stringify(allData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `performance-charts-${this.currentDate.toISOString().slice(0, 10)}.json`);
    linkElement.click();
  }

  // Refresh data (simulated)
  refreshData(): void {
    this.isLoading.set(true);
    
    // Simulate API call
    setTimeout(() => {
      // Update some random data points
      this.performanceCharts.update(charts => 
        charts.map(chart => ({
          ...chart,
          p5Value: Math.max(0, chart.p5Value + (Math.random() - 0.5) * 10),
          p50Value: Math.max(0, chart.p50Value + (Math.random() - 0.5) * 8),
          p95Value: Math.max(0, chart.p95Value + (Math.random() - 0.5) * 6)
        }))
      );
      
      this.isLoading.set(false);
    }, 1000);
  }

  // Get color for value based on thresholds
  getValueColor(value: number, chartType: string): string {
    switch (chartType) {
      case 'processing-time':
        return value < 100 ? '#2ecc71' : value < 200 ? '#f39c12' : '#e74c3c';
      case 'memory-usage':
        return value < 1500 ? '#2ecc71' : value < 2000 ? '#f39c12' : '#e74c3c';
      case 'detection-accuracy':
        return value >= 95 ? '#2ecc71' : value >= 90 ? '#f39c12' : '#e74c3c';
      default:
        return '#3498db';
    }
  }

  // Calculate chart statistics
  getChartStats(chart: PerformanceChart) {
    const values = chart.data.map(d => d.value);
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      stdDev: this.calculateStdDev(values)
    };
  }

  private calculateStdDev(values: number[]): number {
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
    return Math.sqrt(avgSquareDiff);
  }

  // Method to generate SVG path for line/area charts
  getLinePath(chart: PerformanceChart): string {
    if (chart.data.length === 0) return '';
    
    const stats = this.getChartStats(chart);
    const points = chart.data.map((point, index) => {
      const x = (index / (chart.data.length - 1)) * 380 + 10;
      const y = 200 - (point.value / stats.max) * 180 - 10;
      return `${x},${y}`;
    });
    
    return points.join(' ');
  }

  // Get scatter point X position (no Math.random in template)
  getScatterPointX(index: number, totalPoints: number): number {
    // Evenly distribute points with a little randomness for visual appeal
    const basePosition = (index / totalPoints) * 380 + 10;
    const randomOffset = (Math.random() - 0.5) * 30; // ±15 units
    return basePosition + randomOffset;
  }

  // Get scatter point Y position based on value
  getScatterPointY(value: number, maxValue: number): number {
    return 200 - (value / maxValue) * 180 - 10;
  }

  // Navigation
  nextChart(): void {
    const charts = this.filteredCharts();
    const currentIndex = charts.findIndex(c => c.id === this.selectedChartId());
    const nextIndex = (currentIndex + 1) % charts.length;
    this.selectedChartId.set(charts[nextIndex]?.id || charts[0]?.id);
  }

  prevChart(): void {
    const charts = this.filteredCharts();
    const currentIndex = charts.findIndex(c => c.id === this.selectedChartId());
    const prevIndex = (currentIndex - 1 + charts.length) % charts.length;
    this.selectedChartId.set(charts[prevIndex]?.id || charts[0]?.id);
  }

  // Round numbers (replaces Math.round in template)
  roundNumber(value: number): number {
    return Math.round(value);
  }
}