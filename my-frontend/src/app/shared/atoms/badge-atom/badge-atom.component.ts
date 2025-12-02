import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-badge-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './badge-atom.component.html',
  styleUrls: ['./badge-atom.component.scss']
})
export class BadgeAtomComponent {

  @Input() label: string = '';
  @Input() color: string = '#E5E7EB'; // light gray background
  @Input() textColor: string = '#111'; // black text

  @Input() size: 'sm' | 'md' | 'lg' = 'md';

  @Input() rounded: boolean = false; // pill or normal shape

  get classes() {
    return {
      ['badge-' + this.size]: true,
      ['rounded-pill']: this.rounded,
    };
  }
}
