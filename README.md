# StepMania Converter
A file converter for .scc files (StepMania) to .nte files for FlyRanking a870ii dance pad mats. This software may not fully work and is beta software. Contributions are appreciated.<br>

!!! YOU MUST GET YOUR OWN MP3S !!!

### Using SMC
Download .py files<br>
Download any SCC file you want, with the MP3 you want to go with it.<br>
Download the starlevel.dat from HERE for patching.<br>

Run this to convert the .scc ```python3 ssc2nte_a870ii.py "[file].ssc" "[file].nte" --chart Hard --fps 48 --length-s 120```.<br>

Run this to watch the starlevel: ```python3 starlevel_add_song.py starlevel.dat starlevel_patched.dat --title "My Custom Song" --slot 9```.<br>
Or replace a song with: ```python3 starlevel_add_song.py starlevel.dat starlevel_patched.dat --title "My Custom Song" --template 143```.<br>
