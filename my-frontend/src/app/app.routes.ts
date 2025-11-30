import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: 'play',
    loadComponent: () =>
      import('./playground/playground.page').then(m => m.PlaygroundPage),
  },
  {
    path: '',
    redirectTo: 'play',
    pathMatch: 'full',
  },
  {
    path: '**',
    redirectTo: 'play',
  },
];
