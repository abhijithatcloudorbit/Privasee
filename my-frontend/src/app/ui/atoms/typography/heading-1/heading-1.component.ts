import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-heading-1',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './heading-1.component.html',
  styleUrls: ['./heading-1.component.scss']
})
export class Heading1Component {
  @Input() class = '';
}