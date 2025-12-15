import { Component, signal, inject, OnInit } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs';
import { NavTabsComponent } from '../../molecules/navigation/nav-tabs/nav-tabs.component';
import { CommonModule } from '@angular/common';

export interface NavigationItem {
  label: string;
  value: string;
  path: string;
  icon: string;
  disabled?: boolean;
}

@Component({
  selector: 'app-navigation',
  standalone: true,
  imports: [CommonModule, NavTabsComponent],
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.scss']
})
export class NavigationComponent implements OnInit {
  private router = inject(Router);
  
  // Signal-based state management (consistent with project)
  navigationItems = signal<NavigationItem[]>([
    { label: 'Upload', value: 'upload', path: '/upload', icon: '📤' },
    { label: 'Processing', value: 'processing', path: '/processing', icon: '🎨' },
    { label: 'Compliance', value: 'compliance', path: '/compliance', icon: '📊', disabled: true },
    { label: 'Settings', value: 'settings', path: '/settings', icon: '⚙️', disabled: true }
  ]);
  
  currentPath = signal<string>('/upload');
  isLoading = signal<boolean>(false);

  ngOnInit(): void {
    // Set initial path based on current route
    this.currentPath.set(this.router.url);
    
    // Listen for route changes to update active state
    this.router.events
      .pipe(filter(event => event instanceof NavigationEnd))
      .subscribe((event: NavigationEnd) => {
        this.currentPath.set(event.urlAfterRedirects);
        this.handleRouteChange(event.urlAfterRedirects);
      });
  }

  handleNavigation(path: string): void {
    const item = this.navigationItems().find(item => item.path === path);
    
    if (item?.disabled) {
      console.warn(`Navigation to ${item.label} is disabled`);
      return;
    }
    
    this.isLoading.set(true);
    this.currentPath.set(path);
    
    // Navigate to the selected path
    this.router.navigate([path]).then(() => {
      this.isLoading.set(false);
    }).catch(error => {
      console.error('Navigation failed:', error);
      this.isLoading.set(false);
    });
  }

  private handleRouteChange(path: string): void {
    // Update any state based on route change
    console.log('Route changed to:', path);
    
    // Example: Enable compliance route if we're on processing page
    if (path === '/processing') {
      this.navigationItems.update(items => 
        items.map(item => 
          item.value === 'compliance' ? { ...item, disabled: false } : item
        )
      );
    }
  }

  // Helper method to check if a route is active
  isRouteActive(path: string): boolean {
    return this.currentPath() === path;
  }
}