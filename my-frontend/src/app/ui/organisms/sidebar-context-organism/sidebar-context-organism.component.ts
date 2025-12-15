// sidebar-context-organism.component.ts
import { Component, signal, computed, ChangeDetectionStrategy, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// Interfaces
interface SidebarContext {
  id: string;
  title: string;
  icon: string;
  description: string;
  toolType: 'selection' | 'brush' | 'eraser' | 'text_tool' | 'object_tool' | 'compliance' | 'analytics';
  isActive: boolean;
  order: number;
  metadata?: Record<string, any>;
}

interface ToolPreset {
  id: string;
  label: string;
  icon: string;
  value: any;
  category: string;
  description?: string;
}

interface BrushToolSettings {
  size: number;
  opacity: number;
  hardness: number;
  flow: number;
  spacing: number;
  roundness: number;
  angle: number;
  smoothing: number;
  pressureSensitivity: boolean;
  preset: string;
}

interface SelectionToolSettings {
  mode: 'rectangle' | 'ellipse' | 'lasso' | 'polygon' | 'magic_wand';
  feather: number;
  antiAlias: boolean;
  expand: number;
  contract: number;
  border: number;
}

interface TextToolSettings {
  fontFamily: string;
  fontSize: number;
  fontWeight: 'normal' | 'bold' | 'light';
  color: string;
  alignment: 'left' | 'center' | 'right' | 'justify';
  lineHeight: number;
  letterSpacing: number;
}

interface ObjectDetectionSettings {
  confidenceThreshold: number;
  iouThreshold: number;
  detectionTypes: string[];
  autoApply: boolean;
  showBoundingBoxes: boolean;
  showLabels: boolean;
  showConfidence: boolean;
}

@Component({
  selector: 'app-sidebar-context-organism',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './sidebar-context-organism.component.html',
  styleUrls: ['./sidebar-context-organism.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SidebarContextOrganismComponent {
  // ========== INPUT SIGNALS ==========
  @Input() set activeTool(value: string) {
    this.currentTool.set(value);
    this.updateContext(value);
  }
  
  @Input() set imageMetadata(value: any) {
    this.currentImageMetadata.set(value);
  }
  
  @Input() set complianceRequirements(value: string[]) {
    this.activeComplianceRules.set(value);
  }
  
  @Input() set detectionResults(value: any[]) {
    this.currentDetections.set(value);
  }
  
  // ========== OUTPUT EMITTERS ==========
  @Output() toolSettingsChanged = new EventEmitter<any>();
  @Output() toolPresetApplied = new EventEmitter<ToolPreset>();
  @Output() quickActionTriggered = new EventEmitter<{action: string, payload?: any}>();
  @Output() contextSwitched = new EventEmitter<string>();
  
  // ========== STATE SIGNALS ==========
  currentTool = signal<string>('select');
  currentContext = signal<SidebarContext | null>(null);
  currentImageMetadata = signal<any>(null);
  currentDetections = signal<any[]>([]);
  activeComplianceRules = signal<string[]>([]);
  
  // ========== TOOL SETTINGS SIGNALS ==========
  brushSettings = signal<BrushToolSettings>({
    size: 20,
    opacity: 100,
    hardness: 85,
    flow: 100,
    spacing: 25,
    roundness: 100,
    angle: 0,
    smoothing: 10,
    pressureSensitivity: false,
    preset: 'default'
  });
  
  selectionSettings = signal<SelectionToolSettings>({
    mode: 'rectangle',
    feather: 0,
    antiAlias: true,
    expand: 0,
    contract: 0,
    border: 0
  });
  
  textToolSettings = signal<TextToolSettings>({
    fontFamily: 'Inter',
    fontSize: 16,
    fontWeight: 'normal',
    color: '#000000',
    alignment: 'left',
    lineHeight: 1.5,
    letterSpacing: 0
  });
  
  detectionSettings = signal<ObjectDetectionSettings>({
    confidenceThreshold: 0.6,
    iouThreshold: 0.5,
    detectionTypes: ['face', 'text', 'license_plate', 'personal_document'],
    autoApply: true,
    showBoundingBoxes: true,
    showLabels: true,
    showConfidence: true
  });
  
  // ========== CONTEXT DATA ==========
  sidebarContexts = signal<SidebarContext[]>([
    {
      id: 'selection_tools',
      title: 'Selection Tools',
      icon: '🔲',
      description: 'Tools for selecting areas of the image',
      toolType: 'selection',
      isActive: true,
      order: 1,
      metadata: { hotkey: 'V', category: 'basic' }
    },
    {
      id: 'brush_tools',
      title: 'Brush & Painting',
      icon: '🖌️',
      description: 'Manual redaction and painting tools',
      toolType: 'brush',
      isActive: false,
      order: 2,
      metadata: { hotkey: 'B', category: 'redaction' }
    },
    {
      id: 'text_tools',
      title: 'Text Tools',
      icon: '📝',
      description: 'Add and edit text annotations',
      toolType: 'text_tool',
      isActive: false,
      order: 3,
      metadata: { hotkey: 'T', category: 'annotation' }
    },
    {
      id: 'object_tools',
      title: 'Object Detection',
      icon: '🎯',
      description: 'Configure AI detection settings',
      toolType: 'object_tool',
      isActive: false,
      order: 4,
      metadata: { hotkey: 'O', category: 'ai' }
    },
    {
      id: 'compliance_tools',
      title: 'Compliance Tools',
      icon: '⚖️',
      description: 'Compliance check and validation tools',
      toolType: 'compliance',
      isActive: false,
      order: 5,
      metadata: { hotkey: 'C', category: 'compliance' }
    },
    {
      id: 'analytics_tools',
      title: 'Image Analytics',
      icon: '📊',
      description: 'Image analysis and metrics',
      toolType: 'analytics',
      isActive: false,
      order: 6,
      metadata: { hotkey: 'A', category: 'analysis' }
    }
  ]);
  
  toolPresets = signal<ToolPreset[]>([
    {
      id: 'soft_brush',
      label: 'Soft Redaction',
      icon: '🟣',
      value: { size: 25, hardness: 30, opacity: 90 },
      category: 'brush',
      description: 'Soft-edged brush for subtle redactions'
    },
    {
      id: 'hard_brush',
      label: 'Hard Redaction',
      icon: '🔴',
      value: { size: 15, hardness: 100, opacity: 100 },
      category: 'brush',
      description: 'Hard-edged brush for complete redaction'
    },
    {
      id: 'face_selection',
      label: 'Face Detection',
      icon: '👤',
      value: { detectionTypes: ['face'], confidence: 0.7 },
      category: 'detection',
      description: 'Optimized for facial recognition'
    },
    {
      id: 'text_selection',
      label: 'Text Detection',
      icon: '🔤',
      value: { detectionTypes: ['text'], confidence: 0.5 },
      category: 'detection',
      description: 'Optimized for text recognition'
    },
    {
      id: 'gdpr_compliance',
      label: 'GDPR Preset',
      icon: '🇪🇺',
      value: { redactionLevel: 'strict', auditTrail: true, retention: '30d' },
      category: 'compliance',
      description: 'GDPR-compliant redaction settings'
    },
    {
      id: 'hipaa_compliance',
      label: 'HIPAA Preset',
      icon: '🏥',
      value: { redactionLevel: 'strict', auditTrail: true, encryption: true },
      category: 'compliance',
      description: 'HIPAA-compliant medical data protection'
    }
  ]);
  
  quickActions = signal([
    { id: 'select_all_faces', label: 'Select All Faces', icon: '👥', hotkey: 'Ctrl+Shift+F' },
    { id: 'redact_selected', label: 'Redact Selected', icon: '🚫', hotkey: 'Ctrl+R' },
    { id: 'apply_blur', label: 'Apply Gaussian Blur', icon: '🌫️', hotkey: 'Ctrl+B' },
    { id: 'apply_pixelate', label: 'Apply Pixelation', icon: '🧱', hotkey: 'Ctrl+P' },
    { id: 'export_selection', label: 'Export Selection', icon: '📤', hotkey: 'Ctrl+E' },
    { id: 'validate_compliance', label: 'Validate Compliance', icon: '✅', hotkey: 'Ctrl+V' }
  ]);
  
  // ========== COMPUTED SIGNALS ==========
  
  activeContextId = computed(() => {
    const tool = this.currentTool();
    const context = this.sidebarContexts().find(ctx => 
      ctx.toolType === this.mapToolToContext(tool)
    );
    return context?.id || 'selection_tools';
  });
  
  activeToolPresets = computed(() => {
    const tool = this.currentTool();
    return this.toolPresets().filter(preset => 
      preset.category === this.mapToolToPresetCategory(tool)
    );
  });
  
  relevantQuickActions = computed(() => {
    const tool = this.currentTool();
    return this.quickActions().filter(action => 
      this.isActionRelevant(action.id, tool)
    );
  });
  
  imageStats = computed(() => {
    const metadata = this.currentImageMetadata();
    const detections = this.currentDetections();
    
    if (!metadata) return null;
    
    return {
      dimensions: `${metadata.width} × ${metadata.height}`,
      fileSize: this.formatFileSize(metadata.fileSize),
      format: metadata.format?.toUpperCase(),
      colorSpace: metadata.colorSpace,
      detectionCount: detections.length,
      complianceScore: this.calculateComplianceScore(detections),
      privacyRisk: this.calculatePrivacyRisk(detections)
    };
  });
  
  complianceStatus = computed(() => {
    const detections = this.currentDetections();
    const rules = this.activeComplianceRules();
    
    if (!detections.length || !rules.length) return null;
    
    const status = {
      gdpr: rules.includes('GDPR') ? this.checkGdprCompliance(detections) : null,
      hipaa: rules.includes('HIPAA') ? this.checkHipaaCompliance(detections) : null,
      dpdp: rules.includes('DPDP') ? this.checkDpdpCompliance(detections) : null,
      ccpa: rules.includes('CCPA') ? this.checkCcpaCompliance(detections) : null
    };
    
    return {
      ...status,
      overallCompliant: Object.values(status).every(s => s === null || s.compliant)
    };
  });
  
  // ========== LIFECYCLE & INITIALIZATION ==========
  
  constructor() {
    // Set initial context based on default tool
    this.updateContext(this.currentTool());
  }
  
  // ========== MISSING METHODS (FIXING ERRORS) ==========
  
  getSelectionModeIcon(mode: string): string {
    const icons: Record<string, string> = {
      'rectangle': '⬜',
      'ellipse': '⭕',
      'lasso': '🧵',
      'polygon': '🔺',
      'magic_wand': '🪄'
    };
    return icons[mode] || '🔲';
  }
  
  toggleDetectionType(type: string): void {
    this.detectionSettings.update(settings => {
      const currentTypes = [...settings.detectionTypes];
      const index = currentTypes.indexOf(type);
      
      if (index > -1) {
        currentTypes.splice(index, 1);
      } else {
        currentTypes.push(type);
      }
      
      return { ...settings, detectionTypes: currentTypes };
    });
  }
  
  updateConfidenceThreshold(value: string): void {
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      this.detectionSettings.update(settings => ({
        ...settings,
        confidenceThreshold: numValue
      }));
    }
  }
  
  updateAutoApply(checked: boolean): void {
    this.detectionSettings.update(settings => ({
      ...settings,
      autoApply: checked
    }));
  }
  
  updateShowBoundingBoxes(checked: boolean): void {
    this.detectionSettings.update(settings => ({
      ...settings,
      showBoundingBoxes: checked
    }));
  }
  
  // ========== CONTEXT MANAGEMENT ==========
  
  private updateContext(tool: string): void {
    const contextType = this.mapToolToContext(tool);
    const context = this.sidebarContexts().find(ctx => ctx.toolType === contextType);
    
    if (context) {
      // Update all contexts to set only this one as active
      this.sidebarContexts.update(contexts =>
        contexts.map(ctx => ({
          ...ctx,
          isActive: ctx.id === context.id
        }))
      );
      
      this.currentContext.set(context);
      this.contextSwitched.emit(context.id);
    }
  }
  
  switchContext(contextId: string): void {
    const context = this.sidebarContexts().find(ctx => ctx.id === contextId);
    if (context) {
      this.updateContext(context.toolType);
    }
  }
  
  private mapToolToContext(tool: string): SidebarContext['toolType'] {
    const mapping: Record<string, SidebarContext['toolType']> = {
      'select': 'selection',
      'brush': 'brush',
      'eraser': 'brush', // Eraser uses brush context
      'text': 'text_tool',
      'detect': 'object_tool',
      'compliance': 'compliance',
      'analytics': 'analytics'
    };
    
    return mapping[tool] || 'selection';
  }
  
  private mapToolToPresetCategory(tool: string): string {
    const mapping: Record<string, string> = {
      'select': 'selection',
      'brush': 'brush',
      'eraser': 'brush',
      'detect': 'detection',
      'compliance': 'compliance'
    };
    
    return mapping[tool] || 'general';
  }
  
  private isActionRelevant(actionId: string, tool: string): boolean {
    const relevanceMap: Record<string, string[]> = {
      'select_all_faces': ['select', 'detect'],
      'redact_selected': ['brush', 'select'],
      'apply_blur': ['brush', 'select'],
      'apply_pixelate': ['brush', 'select'],
      'export_selection': ['select', 'analytics'],
      'validate_compliance': ['compliance', 'analytics']
    };
    
    return relevanceMap[actionId]?.includes(tool) || false;
  }
  
  // ========== TOOL SETTINGS METHODS ==========
  
  updateBrushSetting(setting: keyof BrushToolSettings, value: any): void {
    this.brushSettings.update(settings => ({
      ...settings,
      [setting]: value
    }));
    
    this.toolSettingsChanged.emit({
      tool: 'brush',
      settings: this.brushSettings()
    });
  }
  
  updateSelectionSetting(setting: keyof SelectionToolSettings, value: any): void {
    this.selectionSettings.update(settings => ({
      ...settings,
      [setting]: value
    }));
    
    this.toolSettingsChanged.emit({
      tool: 'select',
      settings: this.selectionSettings()
    });
  }
  
  applyToolPreset(presetId: string): void {
    const preset = this.toolPresets().find(p => p.id === presetId);
    if (!preset) return;
    
    // Apply preset based on category
    switch (preset.category) {
      case 'brush':
        this.brushSettings.update(settings => ({
          ...settings,
          ...preset.value
        }));
        break;
      case 'detection':
        this.detectionSettings.update(settings => ({
          ...settings,
          ...preset.value
        }));
        break;
      case 'compliance':
        // Emit compliance preset
        break;
    }
    
    this.toolPresetApplied.emit(preset);
  }
  
  triggerQuickAction(actionId: string): void {
    const action = this.quickActions().find(a => a.id === actionId);
    if (!action) return;
    
    const payload = this.getActionPayload(actionId);
    this.quickActionTriggered.emit({ action: actionId, payload });
  }
  
  private getActionPayload(actionId: string): any {
    switch (actionId) {
      case 'select_all_faces':
        return { detectionType: 'face' };
      case 'redact_selected':
        return { method: 'redact', color: '#000000' };
      case 'apply_blur':
        return { method: 'blur', radius: 15 };
      case 'apply_pixelate':
        return { method: 'pixelate', size: 10 };
      default:
        return null;
    }
  }
  
  // ========== COMPLIANCE METHODS ==========
  
  private checkGdprCompliance(detections: any[]): { compliant: boolean; issues: string[] } {
    const issues: string[] = [];
    
    // Check for facial detection without consent
    const faces = detections.filter(d => d.type === 'face');
    if (faces.length > 0) {
      issues.push(`${faces.length} face(s) detected - GDPR Article 9 requires explicit consent`);
    }
    
    // Check for sensitive personal data
    const sensitiveData = detections.filter(d => 
      ['license_plate', 'id_card', 'passport', 'credit_card'].includes(d.type)
    );
    
    if (sensitiveData.length > 0) {
      issues.push(`${sensitiveData.length} sensitive data point(s) detected`);
    }
    
    return {
      compliant: issues.length === 0,
      issues
    };
  }
  
  private checkHipaaCompliance(detections: any[]): { compliant: boolean; issues: string[] } {
    const issues: string[] = [];
    
    // Check for PHI (Protected Health Information)
    const phiDetections = detections.filter(d => 
      d.type === 'text' && 
      this.isPhiText(d.text)
    );
    
    if (phiDetections.length > 0) {
      issues.push(`${phiDetections.length} PHI element(s) detected - HIPAA §164.512 requires de-identification`);
    }
    
    return {
      compliant: issues.length === 0,
      issues
    };
  }
  
  private checkDpdpCompliance(detections: any[]): { compliant: boolean; issues: string[] } {
    const issues: string[] = [];
    
    // Check for personal data as per DPDPA
    const personalData = detections.filter(d => 
      ['face', 'text'].includes(d.type)
    );
    
    if (personalData.length > 0) {
      issues.push(`${personalData.length} personal data point(s) detected - DPDPA Section 4 requires notice and consent`);
    }
    
    return {
      compliant: issues.length === 0,
      issues
    };
  }
  
  private checkCcpaCompliance(detections: any[]): { compliant: boolean; issues: string[] } {
    const issues: string[] = [];
    
    // Check for personal information under CCPA
    const personalInfo = detections.filter(d => 
      d.type === 'text' && 
      this.isPersonalInfoText(d.text)
    );
    
    if (personalInfo.length > 0) {
      issues.push(`${personalInfo.length} personal information item(s) detected - CCPA §1798.140 requires opt-out options`);
    }
    
    return {
      compliant: issues.length === 0,
      issues
    };
  }
  
  private isPhiText(text: string): boolean {
    const phiKeywords = [
      'patient', 'medical', 'diagnosis', 'treatment', 'hospital',
      'doctor', 'prescription', 'ssn', 'social security', 'insurance',
      'medicare', 'medicaid', 'health', 'clinic', 'pharmacy'
    ];
    
    return phiKeywords.some(keyword => 
      text.toLowerCase().includes(keyword.toLowerCase())
    );
  }
  
  private isPersonalInfoText(text: string): boolean {
    const personalInfoPatterns = [
      /\b\d{3}-\d{2}-\d{4}\b/, // SSN
      /\b\d{3}\s\d{2}\s\d{4}\b/,
      /\b[A-Z]\d{6}\b/, // California DL
      /\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b/, // Credit card
      /\b\d{16}\b/
    ];
    
    return personalInfoPatterns.some(pattern => pattern.test(text));
  }
  
  private calculateComplianceScore(detections: any[]): number {
    if (!detections.length) return 100;
    
    const redacted = detections.filter(d => d.redacted).length;
    return Math.round((redacted / detections.length) * 100);
  }
  
  private calculatePrivacyRisk(detections: any[]): 'low' | 'medium' | 'high' {
    if (!detections.length) return 'low';
    
    const sensitiveCount = detections.filter(d => 
      ['face', 'license_plate', 'id_card', 'passport'].includes(d.type)
    ).length;
    
    const ratio = sensitiveCount / detections.length;
    
    if (ratio > 0.5) return 'high';
    if (ratio > 0.2) return 'medium';
    return 'low';
  }
  
  // ========== UTILITY METHODS ==========
  
  private formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
  
  getContextIcon(toolType: string): string {
    const icons: Record<string, string> = {
      'selection': '🔲',
      'brush': '🖌️',
      'text_tool': '📝',
      'object_tool': '🎯',
      'compliance': '⚖️',
      'analytics': '📊'
    };
    
    return icons[toolType] || '🔧';
  }
  
  getRiskColor(risk: string): string {
    return {
      'low': '#2ecc71',
      'medium': '#f39c12',
      'high': '#e74c3c'
    }[risk] || '#95a5a6';
  }
  
  getComplianceColor(compliant: boolean): string {
    return compliant ? '#2ecc71' : '#e74c3c';
  }
  
  // ========== PUBLIC API ==========
  
  resetToDefaults(): void {
    this.brushSettings.set({
      size: 20,
      opacity: 100,
      hardness: 85,
      flow: 100,
      spacing: 25,
      roundness: 100,
      angle: 0,
      smoothing: 10,
      pressureSensitivity: false,
      preset: 'default'
    });
    
    this.selectionSettings.set({
      mode: 'rectangle',
      feather: 0,
      antiAlias: true,
      expand: 0,
      contract: 0,
      border: 0
    });
    
    this.detectionSettings.set({
      confidenceThreshold: 0.6,
      iouThreshold: 0.5,
      detectionTypes: ['face', 'text', 'license_plate', 'personal_document'],
      autoApply: true,
      showBoundingBoxes: true,
      showLabels: true,
      showConfidence: true
    });
    
    this.toolSettingsChanged.emit({ reset: true });
  }
  
  exportToolSettings(): void {
    const settings = {
      exportDate: new Date().toISOString(),
      brush: this.brushSettings(),
      selection: this.selectionSettings(),
      text: this.textToolSettings(),
      detection: this.detectionSettings(),
      context: this.currentContext()
    };
    
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const link = document.createElement('a');
    link.setAttribute('href', dataUri);
    link.setAttribute('download', `tool-settings-${new Date().toISOString().slice(0, 10)}.json`);
    link.click();
  }
}