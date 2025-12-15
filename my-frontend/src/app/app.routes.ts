import { Routes } from '@angular/router';
import { MainLayoutComponent } from './ui/organisms/layout/main-layout/main-layout.component';

export const routes: Routes = [
  {
    path: '',
    component: MainLayoutComponent,
    children: [
      {
        path: '',
        redirectTo: 'upload',
        pathMatch: 'full'
      },
      {
        path: 'upload',
        loadComponent: () => import('./pages/upload/upload-page/upload-page.component').then(m => m.UploadComponent)
      },
      {
        path: 'processing',
        loadComponent: () => import('./pages/processing/processing.component').then(m => m.ProcessingComponent)
      },
      {
        path: 'compliance',
        loadComponent: () => import('./pages/compliance/components/compliance.component').then(m => m.ComplianceComponent)
      },
      {
        path: 'analytics',
        loadComponent: () => import('./pages/analytics/analytics-page/analytics-page.component').then(m => m.AnalyticsPageComponent)
      },
      {
        path: 'settings',
        loadComponent: () => import('./pages/settings/components/settings.component').then(m => m.SettingsComponent)
      }
    ]
  },
  {
    path: '**',
    redirectTo: 'upload'
  }
];