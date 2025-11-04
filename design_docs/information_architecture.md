# Information Architecture

This document defines the static structure of the vidseq application, mapping how all content and functionality is organized and accessed.

## Sitemap

The hierarchical structure of the application:

```
Home / Start Screen
├── Recent Projects List
├── New Project
│   └── Project Initialization Flow
│       ├── Name Project
│       ├── Choose Project Folder
│       └── Video Pipeline Screen (Videos step)
│           ├── Step Selector / Breadcrumb
│           ├── Video List
│           └── Right Sidebar (Add Videos / Add More Videos)
└── Open Project
    └── Project Folder Picker

Project Screen (context: inside a project)
├── Left Navigation Bar
│   ├── Video Pipeline (Videos / Segmentation / Egocentric Alignment)
│   ├── Dimensionality Reduction
│   ├── Behavior Modeling
│   └── Analyze Behavior Model
│
├── Video Pipeline Screen (unified screen with step selector and view states)
│   ├── Step Selector / Breadcrumb (top of screen, always visible)
│   │   ├── Videos step
│   │   ├── Segmentation step
│   │   └── Egocentric Alignment step
│   │
│   ├── Header / Title Area (below step selector)
│   │   ├── Back Arrow (top-left, visible in Touch-Up View only)
│   │   └── Mode Title (Browse Videos / Touch-Up: Full-Body Segmentation / Touch-Up: Head Segmentation)
│   │
│   ├── List View State (default state)
│   │   ├── Main Content Area
│   │   │   └── Video List (contextual columns based on selected step)
│   │   │       ├── Basic status columns (Videos step)
│   │   │       ├── Confidence metrics columns (Segmentation step)
│   │   │       └── Alignment status columns (Egocentric Alignment step)
│   │   │
│   │   └── Right Sidebar (contextual actions based on selected step)
│   │       ├── Videos step actions:
│   │       │   ├── Add Videos / Add More Videos
│   │       │
│   │       ├── Segmentation step actions:
│   │       │   ├── Automatic Segmentation Prompt Input
│   │       │   ├── Run Automatic Preliminary Segmentation
│   │       │   └── Propagate to All Videos
│   │       │
│   │       └── Egocentric Alignment step actions:
│   │           ├── Crop and Mask
│   │           └── Align Egocentric
│   │       │
│   │       └── (Note: Double-clicking a video opens Touch-Up View; type depends on step)
│   │
│   └── Touch-Up View State (when video selected for touch-up)
│       ├── Main Content Area
│       │   ├── Video Player with Mask Overlay (large, fills main area)
│       │   └── Confidence Timeline (below video player)
│       │
│       └── Right Sidebar (touch-up actions)
│           ├── Touch-Up Tools (Positive Label, Negative Label, etc.)
│           ├── Propagate Touch-Ups
│           └── (Note: Back arrow in header returns to List View)
│
├── Dimensionality Reduction Screen
│   ├── Dimensionality Reduction Controls (TBD)
│   └── Navigation to Behavior Modeling
│
├── Behavior Modeling Screen
│   ├── Target Duration Input
│   ├── Number of Iterations Input
│   ├── Fit Behavior Model
│   └── Navigation to Analyze Behavior Model
│
└── Analyze Behavior Model Screen
    ├── Similarity Dendrogram
    ├── Community-Organized Transition Matrix
    ├── Usage Chart
    ├── Grid Movies
    ├── Trajectory Animations
    └── Other Visualizations
```

## Organization Systems

### Hierarchical Structure

The application uses a primary hierarchical organization based on the pipeline workflow:

1. **Top Level**: Home/Start Screen (project selection)
2. **Second Level**: Project Context (individual project screens)
3. **Third Level**: Pipeline Screens and Shared Components (Video Pipeline Screen with step selector and view states (List View and Touch-Up View), Dimensionality Reduction Screen, Behavior Modeling Screen, Analyze Behavior Model Screen)

### Sequential Flow

The pipeline steps form a sequential workflow where each step typically depends on the previous:
- Videos must be added before Segmentation
- Segmentation must be completed before Egocentric Alignment
- Egocentric Alignment must be completed before Dimensionality Reduction
- And so on through the pipeline

### Matrix Access

Within screens, users can access content through multiple dimensions:
- **Video Pipeline Screen**: Unified screen with two view states (List View and Touch-Up View) that can be switched between
- **List View**: Shows video list with contextual columns and actions that change based on selected step (Videos, Segmentation, or Egocentric Alignment)
- **Touch-Up View**: Shows video player with mask overlay, timeline, and touch-up tools when a video is selected for editing
- **Step Selector**: Allows switching between Videos, Segmentation, and Egocentric Alignment steps within the same screen; when in Touch-Up View, switching steps changes the touch-up type for the same video (full-body ↔ head)
- **View Switching**: Users navigate between List View and Touch-Up View by double-clicking videos (List View → Touch-Up View) or using back arrow in header (Touch-Up View → List View)
- **Video List**: Sortable by various metrics (confidence, status, etc.) with columns that change based on selected step; double-clicking a video opens Touch-Up View with context-dependent behavior (full-body segmentation in Segmentation step, head segmentation in Egocentric Alignment step)
- **Left Navigation Bar**: Provides access to Video Pipeline, Dimensionality Reduction, Behavior Modeling, and Analyze Behavior Model screens
- **Right Sidebar**: Provides step-specific actions (List View) or touch-up tools (Touch-Up View) that change contextually based on view state and selected step

## Labeling Systems

### Navigation Labels

- **Videos**: Clear, descriptive name for the initial data import step
- **Segmentation**: Technical but standard term in the field
- **Egocentric Alignment**: Specific technical term describing the alignment process
- **Dimensionality Reduction**: Standard machine learning term
- **Behavior Modeling**: Descriptive term combining domain (behavior) and action (modeling)
- **Analyze Behavior Model**: Action-oriented label indicating the analytical view

### Action Labels

- **Add Videos / Add More Videos**: Context-aware label that changes based on whether videos exist
- **Run Automatic Preliminary Segmentation**: Descriptive, indicating both automation and preliminary nature
- **Propagate Touch-Ups**: Technical but clear action (appears in Touch-Up View right sidebar)
- **Propagate to All Videos**: Clear scope indicator (appears in List View right sidebar)
- **Crop and Mask**: Two-step process combined into single label
- **Align Egocentric**: Concise action description
- **Back Arrow**: Standard navigation control in header to return from Touch-Up View to List View
- **Fit Behavior Model**: Standard machine learning terminology

### Status and Metric Labels

- **mean confidence**: Statistical descriptor
- **5th percentile confidence**: Statistical descriptor indicating quality threshold
- **touched-up status**: Clear completion indicator

## Navigation Systems

### Global Navigation

The Left Navigation Bar appears on all project screens and provides:
- Access to Video Pipeline (unified screen for Videos/Segmentation/Egocentric Alignment), Dimensionality Reduction, Behavior Modeling, and Analyze Behavior Model screens
- Consistent positioning on the left side across all screens
- Visual indication of current screen

The Step Selector / Breadcrumb appears at the top of the Video Pipeline screen and provides:
- Access to Videos, Segmentation, and Egocentric Alignment steps within the unified Video Pipeline screen
- Visual indication of current step within the video processing pipeline
- Sequential workflow progression indicator
- When in Touch-Up View, switching steps changes the touch-up type for the currently selected video (seamless context switching)

The Header / Title Area appears below the step selector and provides:
- Mode indication showing current view state (Browse Videos / Touch-Up: Full-Body Segmentation / Touch-Up: Head Segmentation)
- Back Arrow (top-left) to return from Touch-Up View to List View
- Video name display when in Touch-Up View

### Local Navigation

Within specific screens, local navigation includes:
- **Video Pipeline Screen - List View**: 
  * Step selector at top allows switching between Videos, Segmentation, and Egocentric Alignment steps
  * Right sidebar provides step-specific actions that change based on selected step
  * Double-clicking a video opens Touch-Up View; the touch-up type depends on the current step (full-body segmentation in Segmentation step, head segmentation in Egocentric Alignment step)
  * Video list is sortable and interactive
- **Video Pipeline Screen - Touch-Up View**:
  * Back arrow in top-left header returns to List View
  * Step selector remains visible and functional; switching steps while in Touch-Up View changes the touch-up type for the same video (full-body ↔ head segmentation)
  * Right sidebar provides touch-up tools and actions
  * Video player and timeline fill main content area
  * Header shows current mode (Touch-Up: Full-Body Segmentation or Touch-Up: Head Segmentation) and video name

### Breadcrumbs

Breadcrumbs are explicit through:
- The Step Selector / Breadcrumb at the top of the Video Pipeline screen showing current step (Videos / Segmentation / Egocentric Alignment)
- The Header / Title Area showing current view mode (Browse Videos / Touch-Up: Full-Body Segmentation / Touch-Up: Head Segmentation)
- The Left Navigation Bar showing current screen (Video Pipeline, Dimensionality Reduction, Behavior Modeling, or Analyze Behavior Model)
- The Back Arrow in the header when in Touch-Up View provides clear navigation back to List View

### Contextual Navigation

Navigation is contextual based on workflow state:
- Steps may be disabled until prerequisites are met
- Navigation hints through popovers guide users to next steps
- "Proceed to..." messaging indicates recommended next actions

## Search Systems

Currently, the application does not include traditional search functionality. Content discovery is primarily through:

### Filtering and Sorting

- **Video Lists**: Sortable columns (confidence metrics, status indicators)
- **Visual Filtering**: Users identify problematic videos by sorting confidence metrics from lowest to highest

### Future Search Considerations

Potential search capabilities to consider:
- Search within video metadata (filenames, paths)
- Filter videos by status or completion state
- Search within behavior model results (syllable names, cluster labels)
- Search within analysis visualizations

## Content Inventory

### Primary Content Types

1. **Videos**: Raw video files added to projects
2. **Segmentation Masks**: Generated and manually corrected segmentation data
3. **Cropped/Masked Videos**: Processed videos from egocentric alignment
4. **Dimensionality Reduction Results**: Reduced-dimensional representations
5. **Behavior Models**: Trained model parameters and outputs
6. **Analysis Visualizations**: Charts, graphs, animations, and movies

### Metadata and Status Information

- Project metadata (name, folder location, creation date)
- Video metadata (filenames, paths, processing status)
- Segmentation metrics (confidence scores, completion status)
- Model parameters (target duration, iterations, hyperparameters)
- Processing timestamps and status updates

### Interactive Elements

- Touch-up tools (Positive Label, Negative Label, etc.)
- Video player controls
- Timeline sliders with confidence overlays
- Input fields (text inputs, file pickers)
- Action buttons (various pipeline operations)

## Conceptual Relationships

### Pipeline Dependencies

```
Videos → Segmentation → Egocentric Alignment → Dimensionality Reduction → Behavior Modeling → Analyze Behavior Model
```

Each step produces outputs that become inputs to the next step.

### List-to-Detail View Relationships

```
List View (many videos) ←→ Touch-Up View (single video)
```

Users move between aggregate list views and individual video touch-up views within the same Video Pipeline screen. The transition is seamless - the video list is replaced by the video player, and the right sidebar switches from step actions to touch-up tools.

### View State Switching

```
Video Pipeline Screen
├── List View State (default)
│   ├── Video List (main area)
│   └── Step Actions (right sidebar)
│
└── Touch-Up View State (when video selected)
    ├── Video Player + Timeline (main area)
    └── Touch-Up Tools (right sidebar)
```

The Video Pipeline Screen uses a view state pattern where the same screen layout is maintained, but the content changes between List View and Touch-Up View. The step selector remains visible in both states, allowing users to switch touch-up types (full-body ↔ head) without returning to the list.

### Unified Screen Pattern with View States

```
Video Pipeline Screen (single screen with contextual views and states)
├── Step Selector (top, always visible) - switches between Videos / Segmentation / Egocentric Alignment
│   └── When in Touch-Up View, switching steps changes touch-up type for same video
│
├── Header / Title Area (below step selector)
│   ├── Back Arrow (top-left, visible in Touch-Up View)
│   └── Mode Title (Browse Videos / Touch-Up: Full-Body / Touch-Up: Head)
│
└── Main Content + Right Sidebar (switches between view states)
    ├── List View:
    │   ├── Video List (main area) - contextual columns based on selected step
    │   └── Step Actions (right sidebar) - contextual actions based on selected step
    │
    └── Touch-Up View:
        ├── Video Player + Timeline (main area) - full screen space for editing
        └── Touch-Up Tools (right sidebar) - editing tools and propagate action
```

The Video Pipeline Screen uses a unified screen pattern with two view states. The step selector determines which step context is active (Videos, Segmentation, or Egocentric Alignment), and the view state determines whether the user is browsing videos (List View) or editing a single video (Touch-Up View). This design maximizes screen space for editing while maintaining clear navigation and context.

### Project-to-Step Relationships

```
Project (container) → Multiple Pipeline Screens (content)
├── Video Pipeline Screen (unified screen with 3 steps: Videos, Segmentation, Egocentric Alignment)
├── Dimensionality Reduction Screen
├── Behavior Modeling Screen
└── Analyze Behavior Model Screen
```

Each project contains all pipeline screens. The Video Pipeline Screen is a unified interface that contains three sequential steps (Videos, Segmentation, Egocentric Alignment) accessed through a step selector, while other pipeline stages (Dimensionality Reduction, Behavior Modeling, Analyze Behavior Model) remain as separate screens accessed via the left navigation bar.

