import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
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
  navItems: NavTabItem[] = [
    { label: 'Images', value: 'images' },
    { label: 'Videos', value: 'videos' },
    { label: 'Documents', value: 'documents' },
  ];

  metrics: Metric[] = [
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
  ];

  onFilesSelected(files: File[]): void {
    // This will be fired from UploadZoneOrganismComponent
    console.log('Files selected in upload page:', files);
  }
}
