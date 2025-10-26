# CCTV Clips Collection Guide

This document helps you find and collect suitable CCTV night accident clips for testing the video accident detection system.

## Quick Search Queries (Copy-paste into YouTube)

Use these exact queries to find CCTV-style accident videos:

### Primary Searches (Night/Low-Light Focus)
- `CCTV car accident night`
- `CCTV traffic accident night`
- `surveillance camera car crash night`
- `intersection crash CCTV night`  
- `traffic collision CCTV surveillance`
- `CCTV road accident compilation`
- `surveillance traffic accident`

### Additional Searches
- `CCTV car crash intersection`
- `street surveillance accident`
- `roadside camera car collision`
- `traffic light CCTV crash`

## What to Look For

### ✅ Good Clips
- **Fixed camera angle** (CCTV/surveillance, not dashcam)
- **Night or low-light conditions** (visible vehicles but challenging lighting)
- **Clear collision events** (10-30 seconds duration)
- **Multiple angles/intersections** (for variety)
- **Non-graphic content** (for professional demos)
- **Short duration** (under 5 minutes total video)

### ❌ Avoid
- Dashcam footage (moving camera)
- Extremely dark/unusable footage
- Graphic injury content
- Copyrighted commercial content
- Very long compilation videos (unless you can trim specific segments)

## How to Add Clips to the Downloader

1. Find a suitable video on YouTube
2. Copy the video ID from the URL:
   - URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
   - Video ID: `dQw4w9WgXcQ`

3. Note the timestamp of the accident event:
   - Start: e.g., 1:15 → 75 seconds
   - End: e.g., 1:35 → 95 seconds

4. Open `scripts/download_cctv_clips.py`

5. Add an entry to the `CLIPS_TO_DOWNLOAD` list:
   ```python
   ("VIDEO_ID", start_sec, end_sec, "clip_name", "description", "lighting"),
   ```

   Example:
   ```python
   ("dQw4w9WgXcQ", 75, 95, "intersection_night_01", "Intersection collision at night", "night"),
   ```

## Sample Clip Collection Structure

Aim for this distribution:
- **8-10 accident clips** (positive examples)
  - 5-6 night/low-light
  - 2-3 daytime (for comparison)
  - Various angles: intersection, highway, urban street
  
- **5-10 normal clips** (negative examples - no accidents)
  - Same lighting/angle distribution
  - Normal traffic flow

## Running the Downloader

After adding video IDs to the script:

```powershell
# Activate venv (if not already active)
& C:/wamp64/www/Accident_Prediction/.venv/Scripts/Activate.ps1

# Run the downloader
python scripts/download_cctv_clips.py
```

Clips will be saved to:
- Raw videos: `data/raw_videos/`
- Trimmed clips: `data/clips/`
- Manifest: `data/manifest.csv`

## YouTube Search Tips

1. Use YouTube filters:
   - Duration: Short (< 4 minutes)
   - Upload date: Last year (for recent footage)

2. Look for:
   - Traffic safety channels
   - Surveillance compilation channels
   - Road safety awareness videos

3. Check video descriptions for:
   - "CCTV footage"
   - "Surveillance camera"
   - Timestamps of specific incidents

## Legal & Ethical Notes

⚠️ **Important:**
- YouTube videos are typically copyrighted
- Use clips for **internal testing only**
- Do not distribute or publish without permission
- For client demos, prefer:
  - Academic datasets with licenses
  - Your own recorded footage
  - Simulated videos (CARLA, etc.)

## Alternative: Public Datasets

If you need licensed data, consider these academic datasets:

- **CADP** (Car Accident Detection and Prediction)
- **UCF-Crime** (includes traffic accidents)
- **DADA-2000** (dashcam/driver attention)
- **Kaggle** (search "traffic accident dataset")

See the main README for dataset links.

## Example Clips to Find

Target scenarios:
1. Intersection T-bone collision (night)
2. Rear-end collision at traffic light (night)
3. Lane change collision (low light)
4. Pedestrian near-miss (night)
5. Multi-vehicle pileup (any lighting)
6. Normal traffic intersection (night - negative example)
7. Highway normal flow (night - negative example)

## Next Steps

Once you have 8-12 clips collected:
1. Verify they play correctly
2. Check the manifest.csv
3. Run the accident detection script:
   ```powershell
   python scripts/video_accident_detector.py --input data/clips/intersection_night_01.mp4
   ```
