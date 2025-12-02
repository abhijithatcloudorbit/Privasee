import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

export interface NavTabItem {
  label: string;
  value: string;
}

@Component({
  selector: 'app-nav-tabs',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './nav-tabs.component.html',
  styleUrls: ['./nav-tabs.component.scss'],
})
export class NavTabsComponent {
  @Input() items: NavTabItem[] = [];
  @Input() activeValue: string | null = null;

  @Output() tabChange = new EventEmitter<string>();

  onTabClick(item: NavTabItem): void {
    this.activeValue = item.value;
    this.tabChange.emit(item.value);
  }

  isActive(item: NavTabItem): boolean {
    return this.activeValue === item.value;
  }
}
