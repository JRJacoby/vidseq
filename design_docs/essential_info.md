# Essential Information by Step

This document lists the absolute minimum information and data that a user needs to accomplish each step in the VidSeq pipeline, without regard to visual layout or UI structure. Think of this as the data requirements and decision points for each step.

## Videos Step

**User needs to know:**
- Which video files are currently in the project (file paths/names)
- Whether each video has been successfully added/loaded

**User needs to provide:**
- File paths or directory paths containing video files to add

**System needs to track:**
- List of video files in the project
- Status of each video (added successfully or not)

---

## Segmentation Step

**User needs to know:**
- Which videos have been segmented
- For each video: segmentation confidence metrics (mean confidence, 5th percentile confidence, etc.)
- For a specific video being edited: the segmentation mask overlaid on the video frames
- For a specific video being edited: confidence values at each frame (to identify low-confidence frames)
- Current frame position in the video

**User needs to provide:**
- Text prompt for automatic segmentation algorithm
- Decision to run automatic segmentation
- For touch-up editing: which frames need correction
- For touch-up editing: corrections to the segmentation mask (positive/negative labels, etc.)
- Decision to propagate touch-ups to surrounding frames
- Decision to propagate corrections to all videos

**System needs to track:**
- Segmentation status per video (not started / automatic segmentation complete / touched up)
- Segmentation masks per video per frame
- Confidence metrics per video
- Which videos have been touched up
- Touch-up corrections applied

---

## Egocentric Alignment Step

**User needs to know:**
- Which videos have been cropped and masked
- Which videos have had head segmentation touch-ups completed
- Which videos have been egocentrically aligned
- For a specific video being edited: the head segmentation mask overlaid on the video frames
- For a specific video being edited: confidence values at each frame for head segmentation
- Current frame position in the video

**User needs to provide:**
- Decision to run crop and mask process
- For head segmentation touch-up editing: which frames need correction
- For head segmentation touch-up editing: corrections to the head segmentation mask
- Decision to propagate head touch-ups to surrounding frames
- Decision to propagate head corrections to all videos
- Decision to run egocentric alignment

**System needs to track:**
- Crop and mask status per video
- Head segmentation status per video
- Head segmentation masks per video per frame
- Head segmentation confidence metrics per video
- Egocentric alignment status per video
- Cropped/masked video files per video
- Egocentrically aligned video files per video

---

## Dimensionality Reduction

**User needs to know:**
- Whether dimensionality reduction has been completed for the project
- (TBD: what parameters/options are available for dimensionality reduction)
- (TBD: what results/visualizations are shown)

**User needs to provide:**
- (TBD: dimensionality reduction parameters/configuration)
- Decision to run dimensionality reduction

**System needs to track:**
- Dimensionality reduction status (not started / in progress / completed)
- Reduced-dimensional representation of the dataset
- (TBD: other dimensionality reduction outputs)

---

## Behavior Modeling

**User needs to know:**
- Whether behavior modeling has been completed for the project
- Current progress/status of model fitting (if in progress)
- Number of iterations completed vs total iterations
- Hyperparameter adjustment status

**User needs to provide:**
- Target duration (expected median duration of a behavior, e.g., 350ms)
- Number of iterations to run
- Decision to start model fitting

**System needs to track:**
- Behavior modeling status (not started / in progress / completed)
- Model parameters (target duration, iterations)
- Fitted model outputs
- Progress metrics during fitting

---

## Analyze Behavior Model

**User needs to know:**
- Similarity relationships between discovered behaviors (dendrogram data)
- Transition probabilities between behaviors (transition matrix data)
- Usage/frequency of each behavior (usage statistics)
- Visual representations of behaviors (grid movies, trajectory animations)
- Other analysis visualizations and metrics

**User needs to provide:**
- (Potentially: filtering/selection of which behaviors to analyze)
- (Potentially: parameter adjustments for visualizations)

**System needs to track:**
- Behavior similarity matrix
- Behavior transition matrix
- Behavior usage statistics
- Generated visualizations and analysis outputs

---

## Summary: Data Requirements

**Per-Video Data:**
- Video files and metadata
- Segmentation masks and confidence metrics
- Head segmentation masks and confidence metrics
- Crop/mask status and outputs
- Egocentric alignment status and outputs

**Project-Level Data:**
- List of videos in project
- Dimensionality reduction results
- Behavior model parameters and fitted model
- Behavior analysis results and visualizations

**Decision Points:**
- Which videos to add
- Which videos need segmentation touch-up (based on confidence)
- Which frames need correction (based on frame-level confidence)
- When to propagate corrections
- When to proceed to next step
- Model fitting parameters
- Analysis visualization preferences

---

## Information by Type: Cross-Step Mapping

This section lists each piece of information and which steps it appears in.

### Video Files and Paths
- Videos Step

### Video Status (Added Successfully)
- Videos Step

### Segmentation Status Per Video
- Segmentation Step

### Segmentation Confidence Metrics Per Video
- Segmentation Step

### Segmentation Masks Per Video Per Frame
- Segmentation Step

### Frame-Level Confidence Values
- Segmentation Step
- Egocentric Alignment Step

### Current Frame Position
- Segmentation Step
- Egocentric Alignment Step

### Head Segmentation Masks Per Video Per Frame
- Egocentric Alignment Step

### Head Segmentation Confidence Metrics Per Video
- Egocentric Alignment Step

### Head Segmentation Status Per Video
- Egocentric Alignment Step

### Crop and Mask Status Per Video
- Egocentric Alignment Step

### Egocentric Alignment Status Per Video
- Egocentric Alignment Step

### Cropped/Masked Video Files
- Egocentric Alignment Step

### Egocentrically Aligned Video Files
- Egocentric Alignment Step

### Dimensionality Reduction Status
- Dimensionality Reduction Step

### Reduced-Dimensional Representation
- Dimensionality Reduction Step

### Behavior Modeling Status
- Behavior Modeling Step

### Model Parameters (Target Duration, Iterations)
- Behavior Modeling Step

### Progress Metrics During Fitting
- Behavior Modeling Step

### Fitted Model Outputs
- Behavior Modeling Step

### Behavior Similarity Matrix
- Analyze Behavior Model Step

### Behavior Transition Matrix
- Analyze Behavior Model Step

### Behavior Usage Statistics
- Analyze Behavior Model Step

### Generated Visualizations
- Analyze Behavior Model Step
