import { Injectable, signal, computed, effect } from '@angular/core';

export interface PerformanceMetrics {
  uploadSpeed: number; // KB/s
  processingThroughput: number; // files/second
  memoryUsage: number; // MB
  cpuLoad: number; // percentage
  networkLatency: number; // ms
  errorRate: number; // percentage
}

@Injectable({
  providedIn: 'root'
})
export class PerformanceStore {
  // Core metrics
  private readonly _metrics = signal<PerformanceMetrics>({
    uploadSpeed: 0,
    processingThroughput: 0,
    memoryUsage: 0,
    cpuLoad: 0,
    networkLatency: 0,
    errorRate: 0
  });
  
  // Historical data for charts
  private readonly _history = signal<PerformanceMetrics[]>([]);
  private readonly MAX_HISTORY_SIZE = 100;
  
  // Computed signals
  readonly metrics = this._metrics.asReadonly();
  readonly history = this._history.asReadonly();
  
  readonly averageUploadSpeed = computed(() => {
    const history = this._history();
    if (history.length === 0) return 0;
    return history.reduce((sum, m) => sum + m.uploadSpeed, 0) / history.length;
  });
  
  readonly averageThroughput = computed(() => {
    const history = this._history();
    if (history.length === 0) return 0;
    return history.reduce((sum, m) => sum + m.processingThroughput, 0) / history.length;
  });
  
  readonly performanceScore = computed(() => {
    const m = this._metrics();
    
    // Calculate a weighted score (0-100)
    const weights = {
      uploadSpeed: 0.25,
      processingThroughput: 0.35,
      memoryUsage: 0.15,
      cpuLoad: 0.15,
      errorRate: 0.1
    };
    
    // Normalize each metric to 0-1 range
    const normalizedUploadSpeed = Math.min(m.uploadSpeed / 1000, 1); // Assuming 1000 KB/s is max
    const normalizedThroughput = Math.min(m.processingThroughput / 10, 1); // Assuming 10 files/s is max
    const normalizedMemory = 1 - Math.min(m.memoryUsage / 100, 1); // Assuming 100MB is max (inverse)
    const normalizedCpu = 1 - Math.min(m.cpuLoad / 100, 1); // Inverse
    const normalizedErrorRate = 1 - Math.min(m.errorRate / 100, 1); // Inverse
    
    // Calculate weighted sum
    const score = 
      normalizedUploadSpeed * weights.uploadSpeed +
      normalizedThroughput * weights.processingThroughput +
      normalizedMemory * weights.memoryUsage +
      normalizedCpu * weights.cpuLoad +
      normalizedErrorRate * weights.errorRate;
    
    return Math.round(score * 100);
  });
  
  // Status indicators
  readonly status = computed(() => {
    const score = this.performanceScore();
    if (score >= 80) return 'excellent';
    if (score >= 60) return 'good';
    if (score >= 40) return 'fair';
    return 'poor';
  });
  
  // Alerts
  readonly alerts = computed(() => {
    const m = this._metrics();
    const alerts = [];
    
    if (m.cpuLoad > 80) {
      alerts.push({
        type: 'warning' as const,
        message: 'High CPU usage detected',
        details: `CPU load at ${m.cpuLoad.toFixed(1)}%`
      });
    }
    
    if (m.memoryUsage > 80) {
      alerts.push({
        type: 'warning' as const,
        message: 'High memory usage',
        details: `${m.memoryUsage.toFixed(1)}MB used`
      });
    }
    
    if (m.errorRate > 5) {
      alerts.push({
        type: 'error' as const,
        message: 'High error rate',
        details: `${m.errorRate.toFixed(1)}% of operations failing`
      });
    }
    
    return alerts;
  });
  
  constructor() {
    // Auto-update metrics periodically
    this.startMonitoring();
    
    // Effect to maintain history
    effect(() => {
      const metrics = this._metrics();
      this._history.update(history => {
        const newHistory = [...history, { ...metrics }];
        if (newHistory.length > this.MAX_HISTORY_SIZE) {
          newHistory.shift();
        }
        return newHistory;
      });
    });
  }
  
  // Public API
  updateMetrics(partialMetrics: Partial<PerformanceMetrics>): void {
    this._metrics.update(current => ({
      ...current,
      ...partialMetrics
    }));
  }
  
  updateUploadSpeed(speed: number): void {
    this.updateMetrics({ uploadSpeed: speed });
  }
  
  updateThroughput(throughput: number): void {
    this.updateMetrics({ processingThroughput: throughput });
  }
  
  updateMemoryUsage(usage: number): void {
    this.updateMetrics({ memoryUsage: usage });
  }
  
  updateCpuLoad(load: number): void {
    this.updateMetrics({ cpuLoad: load });
  }
  
  updateErrorRate(rate: number): void {
    this.updateMetrics({ errorRate: rate });
  }
  
  clearHistory(): void {
    this._history.set([]);
  }
  
  // Private monitoring
  private startMonitoring(): void {
    if (typeof window !== 'undefined' && 'performance' in window) {
      // Monitor memory if available
      if ('memory' in (performance as any)) {
        setInterval(() => {
          const memory = (performance as any).memory;
          if (memory) {
            const usedMB = Math.round(memory.usedJSHeapSize / (1024 * 1024));
            this.updateMemoryUsage(usedMB);
          }
        }, 5000);
      }
      
      // Mock CPU load simulation (in real app, would use web workers or similar)
      setInterval(() => {
        // Simulate CPU load based on active operations
        const mockCpuLoad = Math.min(100, Math.random() * 30 + 10); // 10-40%
        this.updateCpuLoad(mockCpuLoad);
      }, 3000);
    }
  }
  
  // Utility methods
  getMetricsSnapshot(): PerformanceMetrics {
    return { ...this._metrics() };
  }
  
  getHistoryRange(start: number, end: number): PerformanceMetrics[] {
    const history = this._history();
    return history.slice(start, end);
  }
  
  resetMetrics(): void {
    this._metrics.set({
      uploadSpeed: 0,
      processingThroughput: 0,
      memoryUsage: 0,
      cpuLoad: 0,
      networkLatency: 0,
      errorRate: 0
    });
  }
}