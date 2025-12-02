import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-text-input-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './text-input-atom.component.html',
  styleUrls: ['./text-input-atom.component.scss'],
})
export class TextInputAtomComponent {
  @Input() placeholder: string = '';
  @Input() type: string = 'text';
}
