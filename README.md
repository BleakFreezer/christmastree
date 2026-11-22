# 3D ASCII Christmas Tree

An interactive 3D ASCII Christmas tree animation built with **Python**, **Pygame** (`pygame-ce`), and **NumPy**. The tree renders with 3D shading, progressive bottom-to-top assembly, drifting background snow, interactive click-to-cycle themes, and optional background music.

## Features

- **Progressive Build-Up Animation:** Trunk, tiered foliage/needles, spiral garland, ornaments/decorations, top star, and base gifts appear gradually over time while rotating smoothly in 3D.
- **Click-to-Cycle Themes:** Click on the window/tree to switch between color palettes and themed decorations (`Classic Green`, `Pink Sweets`, `White & Red`, `Lightblue Sweets`).
- **Drifting Snow & Lighting:** Multi-character background snow and dynamic shading/twinkle pulses.
- **Background Music Support:** Optional looping playback for `last_christmas.mp3` if placed in the project folder.
- **Keyboard Controls:** Quick pause/resume, restart, and exit shortcuts.

## Requirements

- **Python** 3.10+ (tested with Python 3.15 + `pygame-ce`)
- `numpy`
- `pygame-ce` (or `pygame`)

Install dependencies via terminal:

```bash
pip install numpy pygame-ce
