import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DividerAtomComponent } from './divider-atom.component';

describe('DividerAtomComponent', () => {
  let component: DividerAtomComponent;
  let fixture: ComponentFixture<DividerAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DividerAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DividerAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
