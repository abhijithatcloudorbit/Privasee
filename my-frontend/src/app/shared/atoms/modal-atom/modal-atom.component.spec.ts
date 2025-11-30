import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModalAtomComponent } from './modal-atom.component';

describe('ModalAtomComponent', () => {
  let component: ModalAtomComponent;
  let fixture: ComponentFixture<ModalAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModalAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModalAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
