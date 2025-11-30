import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ToastAtomComponent } from './toast-atom.component';

describe('ToastAtomComponent', () => {
  let component: ToastAtomComponent;
  let fixture: ComponentFixture<ToastAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ToastAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ToastAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
