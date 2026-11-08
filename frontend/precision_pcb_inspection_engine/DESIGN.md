---
name: Precision PCB Inspection Engine
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#434655'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#515f74'
  on-secondary: '#ffffff'
  secondary-container: '#d5e3fd'
  on-secondary-container: '#57657b'
  tertiary: '#ae0010'
  on-tertiary: '#ffffff'
  tertiary-container: '#d52022'
  on-tertiary-container: '#ffecea'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#d5e3fd'
  secondary-fixed-dim: '#b9c7e0'
  on-secondary-fixed: '#0d1c2f'
  on-secondary-fixed-variant: '#3a485c'
  tertiary-fixed: '#ffdad6'
  tertiary-fixed-dim: '#ffb4ab'
  on-tertiary-fixed: '#410002'
  on-tertiary-fixed-variant: '#93000b'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.04em
  code-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  gutter-desktop: 1rem
  margin: 1rem
  margin-desktop: 1.5rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system is engineered for industrial-grade automated optical inspection (AOI) and quality verification of printed circuit boards. It balances technical rigor with visual clarity, enabling hardware engineers, quality technicians, and line operators to identify microscopic flaws at high speed without cognitive fatigue.

The visual style is **Precision Minimalism**:
- **Clarity over ornamentation:** Eliminate decorative elements to maximize data density and visual throughput.
- **Instrument-grade focus:** Clean, clinical, and structured surfaces emphasize inspection imagery, solder analysis data, and diagnostic metrics.
- **Immediate visual hierarchy:** Crisp slate typography combined with decisive diagnostic accents (functional red and amber) guides immediate triage of pass/fail anomalies.
- **Architectural quietness:** High-contrast information sits comfortably on soft, balanced warm-grey and white foundation panels.

## Colors

The palette establishes an ultra-clean optical backdrop punctuated by precise functional colors:

### Canvas & Structural Neutrals
- **Surface Canvas (`#FFFFFF`):** Base layer for high-resolution PCB viewport panels, modal dialogs, and active tool states.
- **Surface Layer 1 (`#F8FAFC`):** Secondary structural background for tool rails, property sheets, and telemetry feeds.
- **Surface Layer 2 (`#F1F5F9`):** Recessed containers, active inspection tracks, and datagrid header rows.
- **Border / Divider (`#E2E8F0`):** Fine 1px boundaries separating modules, camera views, and parameter clusters.
- **Text Primary (`#0F172A`):** Deep charcoal for primary readings, pin numbers, and critical labels.
- **Text Secondary (`#334155`):** Slate tone for metadata, units of measurement, and unselected states.
- **Text Muted (`#64748B`):** Supporting labels, timestamps, and inactive toggle icons.

### Accents & Defect Signaling
- **Primary Control (`#2563EB`):** Royal blue reserved strictly for focal interactive controls, selected bounding boxes, and active tool switches.
- **Critical / Defect (`#DC2626`):** Applied exclusively to failed nets, broken traces, short circuits, and critical alerts.
- **Warning / Tolerance Drift (`#D97706`):** Applied to marginal solder volumes, component misalignment warnings, and threshold shifts.
- **Pass / Nominal (`#16A34A`):** Clean green for verified joints, nominal clearances, and batch clearances.

## Typography

Typography prioritizes micro-legibility and tabular efficiency:
- **Inter** is the core typeface across headlines, metrics, forms, and body copy due to its neutral proportions, distinct number forms, and tall x-height.
- **Tabular Figures:** All measurement values, part coordinates (X, Y, Theta), and error metrics must use tabular numbers (`font-variant-numeric: tabular-nums`) to prevent optical jitter during live continuous scans.
- **JetBrains Mono** is reserved for reference designator IDs (e.g., `R104`, `U12`), gerber coordinate readouts, and terminal logs.
- **Uppercase Labels:** Small labels (`label-sm`) use subtle letter spacing to create distinct column headers and metadata groupings without adding visual weight.

## Layout & Spacing

The layout uses a multi-pane docked interface built for high-resolution displays:
- **Workspace Architecture:** A persistent top diagnostic strip (48px high), a left utility/inspection tool strip, a centered interactive viewport canvas for high-zoom PCB imagery, and a collapsible right-hand telemetry/defect drawer.
- **Grid Discipline:** A compact 8pt base grid rhythm with 4pt micro-steps (`space-xs`) for dense toolbars, inspection lists, and property tables.
- **Fluid Layout Behavior:** Side panels maintain minimum fixed widths (280px to 360px), while the center image viewport stretches fluidly across the remaining area.
- **Spacing Density:** Padding within panels remains compact (`space-sm` to `space-md`) to ensure critical measurement data remains visible above the fold without requiring excessive scrolling.

## Elevation & Depth

Depth is established primarily through clean hairline borders and low-contrast surface levels rather than heavy cast shadows:

- **Surface Tiers:**
  - **Level 0 (Base Canvas):** `#F8FAFC` for application structure and dormant panes.
  - **Level 1 (Card/Panel Surface):** `#FFFFFF` bordered with 1px `#E2E8F0` for active inspectors, inspection queues, and floating tool strips.
  - **Level 2 (Active/Floating Overlays):** `#FFFFFF` paired with an ultra-subtle, non-diffuse ambient shadow (`0 2px 4px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.06)`).
  - **Level 3 (Diagnostic Focus / Modals):** `#FFFFFF` with precise, crisp elevation (`0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.03)`).
- **Outlines over Occlusion:** Avoid muddy shadows beneath inspection tools to maintain clean optical alignment with pixel-level PCB image details.

## Shapes

The design system maintains a calibrated, technical geometry:
- **Base Corner Radius (0.25rem / 4px):** Applied to inputs, buttons, tool items, and data chips. Sharp enough to feel industrial and precise, yet soft enough to avoid harsh visual clipping.
- **Panel Corner Radius (0.5rem / 8px):** Applied to floating tool palettes, image containers, and contextual cards.
- **Zero-Radius Scenarios:** Structural pane splitters, viewport boundaries, and datagrid interior cells use 0px corners to ensure seamless pixel alignment.

## Components

### Buttons
- **Primary:** Solid `#2563EB` background with white text, 4px corner radius, 32px height in standard inspection mode. Hover: `#1D4ED8`. Focus: 2px ring `#93C5FD`.
- **Secondary / Ghost:** Transparent or `#F1F5F9` background, `#0F172A` text, 1px `#E2E8F0` border.
- **Destructive / Flag:** Subtle `#FEF2F2` background with `#DC2626` text and border for defect confirmation actions.

### Inspection Chips & Badges
- **Severity Badges:**
  - **Critical Defect:** `#FEF2F2` background, `#DC2626` text, `#FCA5A5` 1px border.
  - **Warning / Out-of-Spec:** `#FFFBEB` background, `#D97706` text, `#FCD34D` 1px border.
  - **Nominal / Pass:** `#F0FDF4` background, `#16A34A` text, `#86EFAC` 1px border.
- **Geometry:** 20px fixed height, font size 11px semi-bold, uppercase, tabular digits for component counts.

### Data Tables & Component Lists
- **Row Architecture:** 32px compact row height. 1px `#E2E8F0` bottom border.
- **Selected Row:** `#EFF6FF` background with 2px solid `#2563EB` left edge highlight.
- **Column Header:** Sticky top, `#F8FAFC` background, 11px uppercase slate text (`#64748B`), 1px bottom border.

### Input Fields & Value Steppers
- **Height:** 32px.
- **Fill & Stroke:** `#FFFFFF` background, 1px `#E2E8F0` border, `#0F172A` text.
- **Focus State:** 1px `#2563EB` border with 2px `#DBEAFE` ambient outer ring.
- **Numeric Units:** Persistent suffix (e.g., `mm`, `mil`, `μm`) rendered in `#64748B` inside the right margin of the input.

### Defect Bounding Boxes (Canvas Viewport)
- **Defect Region:** 1.5px stroke `#DC2626` with 10% translucent red fill.
- **Warning Region:** 1.5px dashed stroke `#D97706` with 8% translucent amber fill.
- **Active Reticle:** 1px stroke `#2563EB` with corner tick markers.