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
        loadComponent: () => import('./pages/processing/processing-page/processing-page.component').then(m => m.ProcessingPageComponent)
      },
      {
        path: 'compliance',
        loadComponent: () => import('./pages/compliance/compliance-page/compliance-page.component').then(m => m.CompliancePageComponent)
      },
      {
        path: 'analytics',
        loadComponent: () => import('./pages/analytics/analytics-page/analytics-page.component').then(m => m.AnalyticsPageComponent)
      },
      {
        path: 'settings',
        loadComponent: () => import('./pages/settings/settings-page/settings-page.component').then(m => m.SettingsPageComponent)
      }
    ]
  },
  {
    path: '**',
    redirectTo: 'upload'
  }
];