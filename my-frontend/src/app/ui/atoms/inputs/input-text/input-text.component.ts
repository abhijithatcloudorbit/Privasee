import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-input-text',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './input-text.component.html',
  styleUrls: ['./input-text.component.scss']
})
export class InputTextComponent {
  @Input() placeholder = '';
  @Input() value = '';
  @Input() disabled = false;
  @Input() type: 'text' | 'password' | 'email' = 'text';
  @Output() valueChange = new EventEmitter<string>();
}