import { Component, Input, Output, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { NgIf } from '@angular/common';

@Component({
  standalone: true,
  selector: 'app-textarea-atom',
  templateUrl: './textarea-atom.component.html',
  styleUrls: ['./textarea-atom.component.scss'],
  imports: [FormsModule, NgIf],
})
export class TextareaAtomComponent {
  @Input() rows: number = 3;
  @Input() placeholder: string = '';
  @Input() value: string = '';
  @Input() disabled: boolean = false;
  @Input() error: string | null = null;
  @Input() autoResize: boolean = false;   // ← REQUIRED to fix “autoResize not a property”

  @Output() valueChange = new EventEmitter<string>();

  onInput(event: Event) {
    const textarea = event.target as HTMLTextAreaElement;
    this.value = textarea.value;
    this.valueChange.emit(this.value);

    if (this.autoResize) {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }
  }
}
