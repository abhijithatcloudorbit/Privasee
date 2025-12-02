import { TestBed } from '@angular/core/testing';

import { ComplianceManager } from './compliance-manager';

describe('ComplianceManager', () => {
  let service: ComplianceManager;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ComplianceManager);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
