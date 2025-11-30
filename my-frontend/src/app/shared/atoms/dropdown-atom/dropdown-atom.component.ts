import { Component, Input, Output, EventEmitter, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface DropdownOption {
  label: string;
  value: string | number;
  disabled?: boolean;
}

@Component({
  standalone: true,
  selector: 'app-dropdown-atom',
  templateUrl: './dropdown-atom.component.html',
  styleUrls: ['./dropdown-atom.component.scss'],
  imports: [CommonModule],
})
export class DropdownAtomComponent {
  @Input() placeholder = 'Select option';
  @Input() width = '220px';
  @Input() options: DropdownOption[] = [];
  @Input() value: string | number | null = null;
  @Input() disabled = false;

  @Output() valueChange = new EventEmitter<string | number>();

  isOpen = false;
  highlightedIndex = 0;

  /** Getter to safely resolve displayed label */
  get selectedLabel(): string {
    const opt = this.options.find(o => o.value === this.value);
    return opt ? opt.label : this.placeholder;
  }

  toggle() {
    if (this.disabled) return;
    this.isOpen = !this.isOpen;

    if (this.isOpen) {
      const index = this.options.findIndex(o => o.value === this.value);
      this.highlightedIndex = index >= 0 ? index : 0;
    }
  }

  select(option: DropdownOption) {
    if (option.disabled) return;
    this.value = option.value;
    this.valueChange.emit(option.value);
    this.isOpen = false;
  }

  // Close when clicking outside
  @HostListener('document:click', ['$event'])
  closeOnOutsideClick(event: Event) {
    const target = event.target as HTMLElement;
    if (!target.closest('app-dropdown-atom')) {
      this.isOpen = false;
    }
  }

  // Keyboard navigation
  @HostListener('keydown', ['$event'])
  handleKeydown(event: KeyboardEvent) {
    if (!this.isOpen) return;

    switch (event.key) {
      case 'ArrowDown':
        this.highlightedIndex = Math.min(this.highlightedIndex + 1, this.options.length - 1);
        break;

      case 'ArrowUp':
        this.highlightedIndex = Math.max(this.highlightedIndex - 1, 0);
        break;

      case 'Enter': {
        const option = this.options[this.highlightedIndex];
        if (option) this.select(option);
        break;
      }

      case 'Escape':
        this.isOpen = false;
        break;
    }
  }
}
