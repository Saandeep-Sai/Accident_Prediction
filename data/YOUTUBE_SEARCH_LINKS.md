# Sample CCTV Accident Video Sources

## YouTube Search URLs (Click to search)

Copy these URLs and paste into your browser to find suitable clips:

### Night/Low-Light CCTV Accidents
1. https://www.youtube.com/results?search_query=CCTV+car+accident+night
2. https://www.youtube.com/results?search_query=surveillance+camera+car+crash+night
3. https://www.youtube.com/results?search_query=traffic+collision+CCTV+night
4. https://www.youtube.com/results?search_query=intersection+crash+CCTV
5. https://www.youtube.com/results?search_query=CCTV+road+accident+compilation

### General CCTV Traffic Accidents
6. https://www.youtube.com/results?search_query=CCTV+traffic+accident
7. https://www.youtube.com/results?search_query=surveillance+traffic+crash
8. https://www.youtube.com/results?search_query=street+camera+car+accident

### Specific Scenarios
9. https://www.youtube.com/results?search_query=red+light+runner+CCTV
10. https://www.youtube.com/results?search_query=intersection+collision+camera

## Common CCTV Compilation Channels

Look for videos from channels that post traffic safety/surveillance footage:
- Search: "CCTV accidents compilation"
- Search: "traffic camera crashes"
- Search: "surveillance footage accidents"

Filter by:
- Duration: Short (under 4 minutes)
- Sort by: Relevance or View count

## Example Video Structure to Look For

Ideal video characteristics:
- Title contains: "CCTV", "Surveillance", "Camera", "Night"
- Duration: 30 seconds to 3 minutes
- Clear timestamp of incident in description or video
- Fixed camera angle (not dashcam)
- Visible license for educational/research use (rare but check descriptions)

## Quick Start: Manual Collection

1. Click one of the search URLs above
2. Find 2-3 suitable videos
3. For each video:
   - Copy the video ID from URL
   - Note the accident timestamp (watch and record start/end seconds)
   - Add to `scripts/download_cctv_clips.py`

Example:
```
Video URL: https://www.youtube.com/watch?v=ABC123xyz
Video ID: ABC123xyz
Accident at: 0:45 to 1:05 → (45, 65) seconds
```

Add to script:
```python
("ABC123xyz", 45, 65, "example_night_01", "Intersection collision", "night"),
```

## Automated Search Alternative

If you want programmatic search, you can use:
- YouTube Data API v3 (requires API key)
- Google Custom Search (requires API key)
- Manual collection (fastest for small sets)

For this MVP, manual collection of 8-12 clips is fastest and sufficient.

## License Reminder

⚠️ These clips are for **internal testing only**. Do not publish or distribute.
For production/client demos, use:
- Academic datasets (CADP, UCF-Crime)
- Licensed stock footage
- Your own recordings
- Simulated videos (CARLA)
