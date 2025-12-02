import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-file-upload-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './file-upload-card.component.html',
  styleUrls: ['./file-upload-card.component.scss'],
})
export class FileUploadCardComponent {
  @Input() file!: File;
  @Input() index!: number;
}
