import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../atoms/buttons/btn-primary.component';
import { InputFileComponent } from '../../atoms/inputs/input-file/input-file.component';
import { HeadingComponent } from '../../atoms/typography/heading/heading.component';
import { TextComponent } from '../../atoms/typography/text/text.component';
import { BytesPipe } from '../../../pipes/bytes.pipe';

@Component({
  selector: 'app-upload-card',
  standalone: true,
  imports: [
    CommonModule,
    BtnPrimaryComponent,
    InputFileComponent,
    HeadingComponent,
    TextComponent,
    BytesPipe,
  ],
  templateUrl: './upload-card.component.html',
  styleUrls: ['./upload-card.component.scss']
})
export class UploadCardComponent {
  selectedFiles: File[] = [];

  @Output() upload = new EventEmitter<File[]>();

  onFileSelected(files: FileList) {
    this.selectedFiles = Array.from(files);
  }

  onUpload() {
    if (this.selectedFiles.length > 0) {
      this.upload.emit(this.selectedFiles);
    }
  }
}

