import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BadgeAtomComponent } from './badge-atom.component';

describe('BadgeAtomComponent', () => {
  let component: BadgeAtomComponent;
  let fixture: ComponentFixture<BadgeAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BadgeAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BadgeAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
