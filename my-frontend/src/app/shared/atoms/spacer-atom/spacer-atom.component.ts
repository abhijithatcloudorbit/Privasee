import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-spacer-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './spacer-atom.component.html',
  styleUrls: ['./spacer-atom.component.scss']
})
export class SpacerAtomComponent {
  @Input() size: number = 16; // must be used like [size]="24"
  @Input() orientation: 'horizontal' | 'vertical' = 'vertical';
}
