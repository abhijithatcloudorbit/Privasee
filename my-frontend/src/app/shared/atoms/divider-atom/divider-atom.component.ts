import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-divider-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './divider-atom.component.html',
  styleUrls: ['./divider-atom.component.scss']
})
export class DividerAtomComponent {
  @Input() orientation: 'horizontal' | 'vertical' = 'horizontal';
  @Input() thickness: number = 1;              // must use [thickness] in template
  @Input() color: string = '#e5e7eb';
  @Input() length: string = '100%';            // string is fine without brackets
}
