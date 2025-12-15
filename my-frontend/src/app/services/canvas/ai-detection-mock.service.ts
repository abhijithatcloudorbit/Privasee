import { Injectable } from '@angular/core';
import { Detection } from '../../shared/interfaces/canvas.interface';

@Injectable({
  providedIn: 'root'
})
export class AiDetectionMockService {
  private generateId(): string {
    return Math.random().toString(36).substr(2, 9);
  }

  async detectObjects(imageData: string): Promise<Detection[]> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate mock detections
    const detections: Detection[] = [
      {
        id: this.generateId(),
        type: 'face',
        confidence: 0.95,
        bbox: {
          x: 100,
          y: 100,
          width: 80,
          height: 80
        },
        appliedFilter: 'none'
      },
      {
        id: this.generateId(),
        type: 'text',
        confidence: 0.87,
        bbox: {
          x: 200,
          y: 50,
          width: 120,
          height: 30
        },
        metadata: {
          text: 'Sample Text'
        },
        appliedFilter: 'none'
      },
      {
        id: this.generateId(),
        type: 'license_plate',
        confidence: 0.92,
        bbox: {
          x: 300,
          y: 200,
          width: 100,
          height: 40
        },
        appliedFilter: 'none'
      }
    ];

    return detections;
  }

  async detectFaces(imageData: string): Promise<Detection[]> {
    return this.detectObjects(imageData).then(detections =>
      detections.filter(d => d.type === 'face')
    );
  }

  async detectText(imageData: string): Promise<Detection[]> {
    return this.detectObjects(imageData).then(detections =>
      detections.filter(d => d.type === 'text')
    );
  }
}