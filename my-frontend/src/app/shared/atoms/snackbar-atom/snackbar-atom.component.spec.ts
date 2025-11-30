import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SnackbarAtomComponent } from './snackbar-atom.component';

describe('SnackbarAtomComponent', () => {
  let component: SnackbarAtomComponent;
  let fixture: ComponentFixture<SnackbarAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SnackbarAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SnackbarAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
