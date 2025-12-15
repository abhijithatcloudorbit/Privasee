import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output, signal } from '@angular/core';

export interface NavTabItem {
  label: string;
  value: string;
  path: string;
  icon?: string;
}

@Component({
  selector: 'app-nav-tabs',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './nav-tabs.component.html',
  styleUrls: ['./nav-tabs.component.scss'],
})
export class NavTabsComponent {
  // Use signals for state management (consistent with your project)
  activeValue = signal<string | null>(null);
  activePath = signal<string | null>(null);
  
  @Input() set items(value: NavTabItem[]) {
    this.itemsSignal.set(value);
  }
  
  @Input() set initialValue(value: string) {
    this.activeValue.set(value);
  }
  
  @Input() set initialPath(value: string) {
    this.activePath.set(value);
  }
  
  itemsSignal = signal<NavTabItem[]>([]);
  
  @Output() tabChange = new EventEmitter<string>();
  @Output() navChange = new EventEmitter<string>();

  onTabClick(item: NavTabItem): void {
    this.activeValue.set(item.value);
    this.activePath.set(item.path);
    this.tabChange.emit(item.value);
    this.navChange.emit(item.path);
  }

  isActive(item: NavTabItem): boolean {
    return this.activeValue() === item.value || this.activePath() === item.path;
  }
}