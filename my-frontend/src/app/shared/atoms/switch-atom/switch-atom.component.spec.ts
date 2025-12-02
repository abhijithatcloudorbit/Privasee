import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SwitchAtomComponent } from './switch-atom.component';

describe('SwitchAtomComponent', () => {
  let component: SwitchAtomComponent;
  let fixture: ComponentFixture<SwitchAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SwitchAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SwitchAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
