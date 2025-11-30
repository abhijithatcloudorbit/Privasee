import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface TabItem {
  label: string;
  disabled?: boolean;
}

@Component({
  selector: 'app-tabs-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './tabs-atom.component.html',
  styleUrls: ['./tabs-atom.component.scss'],
})
export class TabsAtomComponent {
  /** Array of tab items */
  @Input() tabs: TabItem[] = [];

  /** Active tab index */
  @Input() activeIndex: number = 0;

  /** Emits when tab changes */
  @Output() activeIndexChange = new EventEmitter<number>();

  /** Toggle tab */
  selectTab(index: number) {
    if (this.tabs[index]?.disabled) return;

    this.activeIndex = index;
    this.activeIndexChange.emit(index);
  }
}
