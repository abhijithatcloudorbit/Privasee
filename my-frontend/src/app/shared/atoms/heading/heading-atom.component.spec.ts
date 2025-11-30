import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HeadingAtomComponent } from './heading-atom.component';

describe('HeadingAtomComponent', () => {
  let component: HeadingAtomComponent;
  let fixture: ComponentFixture<HeadingAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HeadingAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HeadingAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
