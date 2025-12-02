import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Output } from '@angular/core';
import { InputFileComponent } from '../../atoms/inputs/input-file/input-file.component';
import { FileUploadCardComponent } from '../../molecules/file-upload-card/file-upload-card.component';

@Component({
  selector: 'app-upload-zone-organism',
  standalone: true,
  imports: [CommonModule, InputFileComponent, FileUploadCardComponent],
  templateUrl: './upload-zone-organism.component.html',
  styleUrls: ['./upload-zone-organism.component.scss'],
})
export class UploadZoneOrganismComponent {
  files: File[] = [];
  isUploading = false;

  @Output() filesSelected = new EventEmitter<File[]>();

  // Called by <app-input-file>
  onInputFilesSelected(fileList: FileList): void {
    this.files = Array.from(fileList);
    this.filesSelected.emit(this.files);
  }

  onUpload(): void {
    if (!this.files.length || this.isUploading) return;

    this.isUploading = true;

    // TODO: replace this with real upload logic / service call
    setTimeout(() => {
      this.isUploading = false;
      // Optionally clear files after upload:
      // this.files = [];
    }, 1000);
  }
}
