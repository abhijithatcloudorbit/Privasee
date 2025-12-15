import { CommonModule } from '@angular/common';
import { Component, signal } from '@angular/core';
import { NavTabsComponent, NavTabItem } from '../../../ui/molecules/navigation/nav-tabs/nav-tabs.component';
import { MetricsPreviewCardComponent } from '../../../ui/molecules/cards/metrics-card/metrics-card.component';
import { UploadZoneOrganismComponent } from '../../../ui/organisms/upload-zone-organism/upload-zone-organism.component';

interface Metric {
  label: string;
  value: string | number;
  trend: number;
  trendPositive: boolean;
  description: string;
}

@Component({
  selector: 'app-upload-page',
  standalone: true,
  imports: [
    CommonModule,
    NavTabsComponent,
    MetricsPreviewCardComponent,
    UploadZoneOrganismComponent,
  ],
  templateUrl: './upload-page.component.html',
  styleUrls: ['./upload-page.component.scss'],
})
export class UploadComponent {
  // Signal-based state management
  navItems = signal<NavTabItem[]>([
    { label: 'Images', value: 'images', path: '/upload/images' },
    { label: 'Videos', value: 'videos', path: '/upload/videos' },
    { label: 'Documents', value: 'documents', path: '/upload/documents' },
  ]);

  metrics = signal<Metric[]>([
    {
      label: 'Total files processed',
      value: 124,
      trend: 18.2,
      trendPositive: true,
      description: 'Across all workspaces this week',
    },
    {
      label: 'PII elements detected',
      value: 452,
      trend: -3.5,
      trendPositive: false,
      description: 'Compared to last week',
    },
    {
      label: 'Compliance score',
      value: '98.4%',
      trend: 2.1,
      trendPositive: true,
      description: 'Of all processed batches',
    },
  ]);

  activeTab = signal<string>('images');

  // Helper method to get accepted file types based on active tab
  getAcceptTypes(): string {
    switch (this.activeTab()) {
      case 'images':
        return 'image/*,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff,.svg';
      case 'videos':
        return 'video/*,.mp4,.mov,.avi,.webm,.mkv,.flv,.wmv,.m4v';
      case 'documents':
        return '.pdf,.doc,.docx,.txt,.rtf,.odt,.pages,.md';
      default:
        return '*/*';
    }
  }

  // Get maximum file size based on active tab
  getMaxFileSize(): number {
    switch (this.activeTab()) {
      case 'images':
        return 10 * 1024 * 1024; // 10MB
      case 'videos':
        return 100 * 1024 * 1024; // 100MB
      case 'documents':
        return 5 * 1024 * 1024; // 5MB
      default:
        return 10 * 1024 * 1024; // 10MB default
    }
  }

  // Get allowed file count based on active tab
  getMaxFileCount(): number {
    switch (this.activeTab()) {
      case 'images':
        return 50;
      case 'videos':
        return 10;
      case 'documents':
        return 20;
      default:
        return 20;
    }
  }

  onFilesSelected(files: File[]): void {
    console.log('Files selected in upload page:', files);
    this.updateMetrics(files.length);
  }

  onTabChange(tabValue: string): void {
    this.activeTab.set(tabValue);
    console.log('Active tab changed to:', tabValue);
  }

  private updateMetrics(newFileCount: number): void {
    this.metrics.update(currentMetrics => {
      const updatedMetrics = [...currentMetrics];
      const totalFilesMetric = updatedMetrics[0];
      
      if (typeof totalFilesMetric.value === 'number') {
        totalFilesMetric.value = totalFilesMetric.value + newFileCount;
        totalFilesMetric.trend = totalFilesMetric.trend > 0 
          ? totalFilesMetric.trend + 0.5 
          : 5.0;
      }

      return updatedMetrics;
    });
  }

  // Get metrics as array for template
  getMetricsArray(): Metric[] {
    return this.metrics();
  }
}