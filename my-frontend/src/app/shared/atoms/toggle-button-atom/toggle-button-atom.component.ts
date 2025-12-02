import { Component, Input, Output, EventEmitter } from '@angular/core';
import { NgClass, NgFor } from '@angular/common';

interface ToggleOption {
  label: string;
  value: string;
  disabled?: boolean;
}

@Component({
  standalone: true,
  selector: 'app-toggle-button-atom',
  templateUrl: './toggle-button-atom.component.html',
  styleUrls: ['./toggle-button-atom.component.scss'],
  imports: [NgClass, NgFor],
})
export class ToggleButtonAtomComponent {
  @Input() options: ToggleOption[] = [];
  @Input() value: string | null = null;
  @Input() color: string = '#2563eb'; // Selected color
  @Input() disabled: boolean = false;

  @Output() valueChange = new EventEmitter<string>();

  onSelect(option: ToggleOption) {
    if (this.disabled || option.disabled) return;

    this.value = option.value;
    this.valueChange.emit(this.value);
  }
}
