MDF Signal Editor 1.0 - offline tool to modify / add / delete signals in MDF files
===============================================================================

Works fully offline. Reads MDF 3.x and MDF 4.x (.mdf, .mf4). Writes a NEW file
in the same MDF version - the original file is never changed.

author = "Rakesh Kyathsandra Jayaramu(uih15068)"

SETUP (Windows)
---------------
Option A - with Python installed (3.9 or newer):
  1. Double-click run_editor.bat
     (first start installs asammdf + matplotlib; after that no internet needed)


HOW TO USE
----------
1. Open MDF file...        -> all signals are listed (unit, samples, rate, min/max).
                              Use the Search box to filter.
2. Define a change:
     Modify selected signal  - select the signal in the list
     Add new signal          - type the new name; choose the time base
                               ("Same as signal" = copy its timestamps, recommended,
                               or a fixed rate in ms over the whole file duration)
     Delete selected signal
   Enter time (s) and value, one pair per line:
       0 0
       10 1
       58 0
   (also accepted: 0:0; 10:1; 58:0  - or paste two columns from Excel)

   Step (default)  : value is held until the next time -> 0 until 10 s,
                     1 from 10 s, 0 again from 58 s.
   Linear ramp     : values are interpolated between the points.
   Only change from/to (Modify only): changes values only inside that time
                     window, original values are kept outside.

   Preview plot shows original vs new.  Then click "Add to change list".
3. Repeat for as many signals as you need (pending changes list).
4. Generate modified MDF file...  -> choose the output name -> done.

Note: for signals with a value-to-text table (enum), enter the raw numbers.

