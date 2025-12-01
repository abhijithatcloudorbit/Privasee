import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-btn-primary',
  standalone: true,
  imports: [CommonModule],
  templateUrl: 'btn-primary.component.html',
  styleUrls: ['./btn-primary.component.scss']
})
export class BtnPrimaryComponent {
  @Input() disabled = false;
  @Input() loading = false;
  @Input() icon = '';
  @Output() onClick = new EventEmitter<Event>();
}