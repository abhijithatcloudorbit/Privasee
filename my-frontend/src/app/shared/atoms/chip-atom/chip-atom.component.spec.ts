import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChipAtomComponent } from './chip-atom.component';

describe('ChipAtomComponent', () => {
  let component: ChipAtomComponent;
  let fixture: ComponentFixture<ChipAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChipAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChipAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
