"""
File to pre-process the data. 
It loads the metadata file, processes it and saves it to a new file. 
It also loads the XML files, processes them and saves the track objects to a dictionary.
The dictionary is then copied without the score attribute and saved to a pickle file.

This has already been done and the files are available in the repository.
The code is provided for reference.
"""

import pickle
import copy
import utils as u

FILES_DIR = 'arab-andalusian-music'
METADATA_FILE = 'arab-andalusian-music/metadata-all-nawbas.csv'
PROCESSED_METADATA_FILE = 'arab-andalusian-music/metadata_processed.csv'

# Load metadata file and process it
metadata = u.process_metadata(METADATA_FILE)

# Save metadata to new file including index
metadata_file = PROCESSED_METADATA_FILE
metadata.to_csv(metadata_file, index=True)

# Load XML files
xml_files = u.load_files(FILES_DIR)
print(f'Loaded {len(xml_files)} files')

# Process XML files and load track objects into a dictionary
from utils import Track
tracks = {}
for i, xml_file in enumerate(xml_files):
    print(f'Processing file {i+1}/{len(xml_files)}')
    try:
        track_obj = Track(xml_file)
        tracks[track_obj.track_id] = track_obj
    except ValueError as e:
        print(e)

# Create a copy of the tracks dictionary without the score attribute
tracks_without_score = {}
for track_id, track in tracks.items():
    track_copy = copy.copy(track)
    track_copy.score = None
    tracks_without_score[track_id] = track_copy

# Save the copied dictionary to a pickle file
with open('tracks.pkl', 'wb') as f:
    pickle.dump(tracks_without_score, f)