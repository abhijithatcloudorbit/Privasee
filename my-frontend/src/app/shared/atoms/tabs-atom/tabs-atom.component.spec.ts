import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TabsAtomComponent } from './tabs-atom.component';

describe('TabsAtomComponent', () => {
  let component: TabsAtomComponent;
  let fixture: ComponentFixture<TabsAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TabsAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TabsAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
