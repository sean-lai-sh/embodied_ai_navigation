# Embodied AI Navigation

Emulation of scenario where robot needs to go from A to B given ideas of what a "wall" is.

# Instructions for Players
1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Play using the default keyboard player
```commandline
python player.py
```

3. Modify the player.py to implement your own solutions, 
unless you have photographic memories!

# Solution Run down
1. Download exploration data from link provided in the class.
2. Unzip exploration_data.zip and place 'images' and 'images_subsample' under 'vis_nav_player/data' folder.
3. Place 'startup.json' under 'vis_nav_player/' folder.
4. Run baseline code
   ```
   python baseline.py
   ```# embodied_ai_navigation
