import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-heading-2',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './heading-2.component.html',
  styleUrls: ['./heading-2.component.scss']
})
export class Heading2Component {
  @Input() class = '';
}