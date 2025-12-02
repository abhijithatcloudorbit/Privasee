import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  standalone: true,
  selector: 'app-button-atom',
  imports: [CommonModule],
  templateUrl: './button-atom.component.html',
  styleUrls: ['./button-atom.component.scss'],
})
export class ButtonAtomComponent {
  @Input() color: 'primary' | 'secondary' | 'danger' = 'primary';
}
