import { Component, Input, Output, EventEmitter } from '@angular/core';
import { NgClass } from '@angular/common';

@Component({
  standalone: true,
  selector: 'app-switch-atom',
  templateUrl: './switch-atom.component.html',
  styleUrls: ['./switch-atom.component.scss'],
  imports: [NgClass],
})
export class SwitchAtomComponent {
  @Input() checked: boolean = false;          // Current state
  @Input() disabled: boolean = false;         // Disable toggle
  @Input() color: string = '#2563eb';         // ON color

  @Output() checkedChange = new EventEmitter<boolean>();

  toggle() {
    if (this.disabled) return;
    this.checked = !this.checked;
    this.checkedChange.emit(this.checked);
  }
}
