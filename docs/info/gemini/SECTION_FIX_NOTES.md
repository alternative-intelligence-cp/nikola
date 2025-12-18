# SECTION NUMBERING FIX - December 11, 2025

## Problem Identified

Gemini was getting confused because:
- The Table of Contents says "Section 2.2 - Wave Interference Physics"
- But inside the actual file, the content has "## 4.1 Wave Interference"
- The internal section numbers don't match the TOC structure
- This caused Gemini to waste time doing "targeted searches" and failing to find content

## Solution Implemented

### 1. Added Clear Section Markers
Every section now has a prominent marker in the compilation files:

```
================================================================================
SECTION 2.2: 02 Wave Interference Physics
================================================================================

<!-- SOURCE: 02_foundations/02_wave_interference_physics.md -->
```

This makes it crystal clear which TOC section you're looking at, regardless of internal numbering.

### 2. Updated Task Instructions
All task files now include explicit guidance:

```
**IMPORTANT - HOW TO FIND SECTIONS:** 
- Each compilation file has clear section markers like:
  ================================================================================
  SECTION X.Y: Component Name
  ================================================================================
- Use text search (Ctrl+F) for keywords instead of section numbers
- The files have internal numbering that differs from the TOC - IGNORE internal numbers
- Search for concepts like "Wave Interference", "Mamba", "ZeroMQ", etc.
```

### 3. Created Navigation Guide
Added `NAVIGATION_GUIDE.md` with:
- Clear DO/DON'T instructions
- Example search strategies
- Quick reference for which sections are in which files

## Files Updated

1. ✅ `utils/compile_plan.py` - Added section mapping and markers
2. ✅ `utils/create_task.py` - Updated instruction template
3. ✅ All 9 compilation files regenerated with markers
4. ✅ All 15 task files regenerated with new instructions
5. ✅ Created `NAVIGATION_GUIDE.md`

## How This Helps Gemini

**Before:**
- "I need to find Section 4.7 (Physics Oracle)"
- Searches for "4.7" → finds "## 4.7" with wrong content
- Searches for "Section 4.7" → not found
- Concludes sections are non-linear and starts random searching

**After:**
- "I need to find Physics Oracle content"
- Searches for "Physics Oracle" → immediately finds it
- Or sees clear marker: "SECTION X.Y: Physics Oracle"
- Content found in seconds, no confusion

## Verification

Recompiled files now have:
- Size: 489-505 KB (slightly larger due to section markers)
- Clear ==== markers every section
- Consistent TOC alignment
- Easy keyword searching

## Next Steps for You

1. **Replace Gemini's files** with the newly compiled ones
2. **Give her the NAVIGATION_GUIDE.md** or just paste the key points
3. **She should search by keywords** (Wave Interference, Mamba, etc.) not section numbers

This should eliminate the "targeted search" time waste and help her find everything immediately!
