```mermaid
flowchart TD
    A[User selects 'Egocentric Alignment' step in the Video Pipeline screen step selector at the top]
    B[If this is the first time the 'Crop and Mask' button has been available for this project, the user sees a popover pointing to the 'Crop and Mask' button in the right sidebar saying 'The first step in egocentric alignment is centering the video and masking out the background. Click here to begin the process.']
    C[User clicks 'Crop and Mask' button in the right sidebar]
    D[User sees status updates on individual videos as cropping and masking proceeds.]
    E[If this is the first time cropping and masking have completed for this project, the user sees a popover pointing to the first video in the list saying 'Double-click a video to open it in the touch-up view for head segmentation editing'"]
    F[User double-clicks a video from the list]
    G[The Video Pipeline screen switches to Touch-Up View and the user sees the selected video opened with the head segmentation mask visible in the main content area. The header shows 'Touch-Up: Head Segmentation' and the video name. The right sidebar now shows touch-up tools.]
    H["If this is the first time a video has been opened in the Touch-Up View for head segmentation in this project, the user sees a popover pointing to a timeline with a graph overlaid showing segmentation confidence at each frame saying 'Use the slider to find frames in the video where the model has low head segmentation confidence.'"]
    I[The user clicks and drags to a location in the video timeline]
    J["If this is the first time the timeline has been used for head segmentation in this project, the user sees a popover pointing to the touch-up tools in the right sidebar with tools like 'Positive Label' and 'Negative Label' and maybe others saying 'Use the touch-up tools to fix poor head segmentations'"]
    K[User clicks on a touch-up tool in the right sidebar]
    L["If this is the first time a touch-up tool has been selected for head segmentation in this project, the user sees a popover pointing to the video player in the main content area saying 'Click on the video pane to add correction with the touch-up tools.'"]
    M[The user clicks on the video player pane]
    N["If this is the first time the user has used a touch-up tool for head segmentation in this project, the user sees a popover pointing to the 'Propagate Touch-Ups' button in the right sidebar saying 'After you touch-up a handful of frames, click 'Propagate Touch-Ups' to apply your touch-ups to the surrounding frames so that you don't have to fix every frame. Try to fix a few frames in each low-confidence region of the video.'"]
    O["The user finishes touch-ups on a few frames and clicks 'Propagate Touch-Ups' button in the right sidebar"]
    P["If this is the first time touch-ups have been propagated for head segmentation in this project, the user sees a popover pointing to the back arrow in the top-left header saying 'Continue touching-up frames and propagating your touch-ups repeatedly until you are completely happy with the head segmentations in this video. Then, click the back arrow to return to the video list.'"]
    Q["The user finishes touching-up this video and clicks the back arrow in the top-left header"]
    R[The Video Pipeline screen switches back to List View, showing the video list with updated status]
    S["If this is the first time a video has completed the head touch-up process in this project, the user sees a popover pointing to the touched-up status symbol in the video's row and also to the 'Propagate to All Videos' button in the right sidebar saying 'Continue touching up more videos by double-clicking them, and then click 'Propagate to all videos' to apply your corrections to the entire dataset. We recommend touching-up at least 5 videos or 10 percent of your videos, whichever is smaller.'"]
    T["The user touches up more videos by double-clicking them, then clicks 'Propagate to All Videos' button in the right sidebar while in List View"]
    U[The user sees status updates on individual videos as the propagation process takes place.]
    V[If this is the first time propagate to all videos has been completed for the Egocentric Alignment step in this project, the user sees a popover pointing to the 'Align Egocentric' button in the right sidebar saying 'Click here to use your new segmentations to align all frames such that mice are always facing the same direction']
    W[User clicks 'Align Egocentric' button in the right sidebar]
    X[User sees status updates on individual videos as the egocentrically aligned versions of videos are created]
    Y[If this is the first time egocentric alignment has completed in this project, the user sees a popover pointing to 'Dimensionality Reduction' in the left navigation bar saying 'Egocentric Alignment has been completed! Move on to Dimensionality Reduction next']
    Z[User clicks 'Dimensionality Reduction' in the left navigation bar and is taken to the Dimensionality Reduction screen]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> U
    U --> V
    V --> W
    W --> X
    X --> Y
    Y --> Z
```

