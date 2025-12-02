import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IconAtomComponent } from './icon-atom.component';

describe('IconAtomComponent', () => {
  let component: IconAtomComponent;
  let fixture: ComponentFixture<IconAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [IconAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IconAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
