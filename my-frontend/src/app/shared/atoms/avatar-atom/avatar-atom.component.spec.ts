import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AvatarAtomComponent } from './avatar-atom.component';

describe('AvatarAtomComponent', () => {
  let component: AvatarAtomComponent;
  let fixture: ComponentFixture<AvatarAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AvatarAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AvatarAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
