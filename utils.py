"""
Utility functions for the project

This file contains utility functions for the project, including functions to load music21 files, process metadata, and perform statistical analysis.

"""
from music21 import *
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from scipy.special import kl_div

PROCESSED_METADATA_FILE = 'arab-andalusian-music/metadata_processed.csv'


class Track:
    """
    Class for a track

    This class represents a track in the dataset.
    It contains methods to load the track, extract metadata, and compute the probability density function of the MIDI notes.

    Class attributes:
    metadata (pd dataframe): metadata of the tracks
    midi_range (np array): range of MIDI notes from 48 to 83

    Object attributes:
    file_path (str): path to the music21 file
    track_id (str): track id
    score (music21 stream): music21 stream object of the track
    tab (str): tab of the track
    mizan (str): mizan of the track
    note_sequence (list): list of notes in the track
    midi_sequence (list): list of MIDI notes in the track
    pdf (np array): probability density function of the MIDI notes
    note_occurrences (dict): dictionary with MIDI note as key and duration (1 = quarter note)
    weighted_pdf (np array): weighted probability density function of the MIDI notes

    Methods:
    __init__: Initialize a track object
    get_track_id: Get the track id from a file path
    get_score: Get the score of the track
    get_tab: Get the tab of the track
    get_mizan: Get the mizan of the track
    get_note_sequence: Get the note sequence of the track
    get_midi_sequence: Get the MIDI sequence of the track
    get_pdf: Compute the probability density function of the MIDI notes
    get_pdf_and_length: Return the PDF and length of the MIDI sequence
    get_weighted_pdf_and_length: Return the weighted PDF and length of the track
    get_note_occurrences: Compute the duration of each MIDI note in the track
    get_weighted_pdf: Compute the weighted probability density function of the MIDI notes
    plot_pdfs: Plot the PDF and weighted PDF of the track
    pdf_similarity: Compute the similarity between the PDF and weighted PDF of the track
    """
    metadata = pd.read_csv(PROCESSED_METADATA_FILE, header=0, index_col='track_id')
    midi_range = np.arange(48, 83)

    def __init__(self, file_path):
        """
        Initialize a track object

        Parameters:
        file_path (str): path to the music21 file

        Attributes:
        file_path (str): path to the music21 file
        track_id (str): track id
        score (music21 stream): music21 stream object of the track
        tab (str): tab of the track
        mizan (str): mizan of the track
        note_sequence (list): list of notes in the track
        """
        self.file_path = file_path
        self.track_id = self.get_track_id(file_path)

        # If track id is not found in metadata, raise an error
        if self.track_id not in self.metadata.index:
            raise ValueError(f'Track id {self.track_id} not found in metadata')
        
        self.score = self.get_score(file_path)
        self.tab = self.get_tab()
        self.mizan = self.get_mizan()
        self.note_sequence = self.get_note_sequence()
        self.midi_sequence = self.get_midi_sequence()
        self.pdf = self.get_pdf()
        self.note_occurences = self.get_note_occurrences()
        self.weighted_pdf = self.get_weighted_pdf()

    def get_track_id(self, file_path):
        """
        Get the track id from a file path

        Parameters:
        file_path (str): path to the file

        Returns:
        track_id (str): track id
        """
        # Get the track id from the file path
        track_id = os.path.basename(file_path).split('.')[0]

        return track_id

    def get_score(self, file_path):
        """
        Get the score of the track

        Parameters:
        file_path (str): path to the music21 file

        Returns:
        score (music21 stream): music21 stream object of the track
        """
        # Load the score of the track
        score = converter.parse(file_path)

        return score
    
    def get_tab(self):
        """
        Get the tab of the track

        Returns:
        tab (str): tab of the track
        """
        # Get the tab of the track
        tab = self.metadata.loc[self.track_id, 'tab']

        return tab

    def get_mizan(self):
        """
        Get the mizan of the track

        Returns:
        mizan (str): mizan of the track
        """
        # Get the mizan of the track
        mizan = self.metadata.loc[self.track_id, 'mizan']

        return mizan
    
    def get_note_sequence(self):
        """
        Get the note sequence of the track

        Returns:
        notes (list): list of notes in the track
        """
        elements = self.score.flat.notes
        # If note is a note, get the name with octave, if chord, ignore the note
        notes = [element.pitch.nameWithOctave for element in elements if isinstance(element, note.Note)]

        return notes

    def get_midi_sequence(self):
        """
        Get the midi sequence of the track, using the note sequence
        Returns:
        midi_sequence (list): list of midi notes in the track
        """
        # Get the midi sequence of the track
        midi_sequence = [pitch.Pitch(note).midi for note in self.note_sequence]

        return midi_sequence

    def get_pdf(self):
        """
        Compute the probability density function of the midi notes, within a midi range
    
        """
        # Get the midi sequence of the track
        midi_sequence = self.midi_sequence
        midi_range = self.midi_range

        # Compute the probability density function of the midi notes, within a midi range
        hist,_ = np.histogram(midi_sequence, bins=np.append(midi_range, max(midi_range)+1), density=True)

        return hist

    def get_pdf_and_length(self):
        """
        Return the pdf and length of the midi sequence

        Returns:
        pdf (np array): pdf of the midi sequence
        length (int): length of the midi sequence
        """
        return self.pdf, len(self.midi_sequence)
    
    def get_weighted_pdf_and_length(self):
        """
        Return the weighted pdf and length track in quarter notes

        Returns:
        weighted_pdf (np array): weighted pdf of the midi notes, within a midi range
        length (int): length of the track in quarter notes
        """
        length = sum(self.note_occurences.values())
        return self.weighted_pdf, length
    
    def get_note_occurrences(self):
        """
        Compute the duration of each MIDI note in the track

        Returns:
        note_occurrences (dict): dictionary with MIDI note as key and duration (1 = quarter note)
        """
        # Initialize dictionary to store MIDI note occurrences
        note_occurrences = {}

        # Iterate over all notes in the score
        for element in self.score.flat.notes:

            # Get the MIDI number and duration of the note, discard chords
            if isinstance(element, note.Note):
                midi_number = element.pitch.midi
                duration = element.duration.quarterLength

                # Adjust the occurrence of each MIDI note by its duration
                if midi_number in note_occurrences:
                    note_occurrences[midi_number] += duration
                else:
                    note_occurrences[midi_number] = duration
        
        return note_occurrences

    def get_weighted_pdf(self):

        """
        Compute the probability density function of the midi notes, taking into account note length

        Returns:
        weighted_pdf (np array): weighted pdf of the midi notes, within a midi range
        """         
        # Get the note occurrences
        note_occurrences = self.get_note_occurrences()

        # Convert note occurrences to a numpy array, based on midi_range
        weighted_pdf = np.array([note_occurrences.get(midi, 0) for midi in self.midi_range])

        # Normalize the weighted pdf
        weighted_pdf = weighted_pdf / np.sum(weighted_pdf)

        return weighted_pdf

    def plot_pdfs(self):
        """
        Plot the PDF and weighted PDF of the track

        Returns:
        None

        NOTE: This function is provided to visualize the PDF and weighted PDF of a track
        """
        # Create a DataFrame with the midi_range, PDF, and weighted PDF
        data = np.vstack([self.midi_range, self.pdf, self.weighted_pdf])
        df = pd.DataFrame(data.T)

        # Rename the columns
        df.columns = ['midi_range', 'PDF', 'Weighted PDF']

        # Convert midi_range to integer
        df['midi_range'] = df['midi_range'].astype(int)

        # Min and max note in the track
        min_note = min(self.midi_sequence)
        max_note = max(self.midi_sequence)

        # Filter the DataFrame to only include data within the range of the track
        df = df[(df['midi_range'] >= min_note) & (df['midi_range'] <= max_note)]

        # Melt the DataFrame to make it suitable for Seaborn
        df_melted = df.melt(id_vars='midi_range', var_name='Variable', value_name='Value')

        # Create the double bar plot using Seaborn
        sns.barplot(data=df_melted, x='midi_range', y='Value', hue='Variable')

        # Plot values
        plt.title(f"{self.track_id} \n {self.tab} {self.mizan}")
        plt.xlabel('MIDI pitch')
        plt.ylabel('Probability')
        plt.legend(title=None)
        plt.xticks(rotation=45)  

        # Display the plot
        plt.show()

    def pdf_similarity(self):
        """
        Compute the similarity between the PDF and weighted PDF of the track

        Returns:
        cos_sim (float): cosine similarity between the PDF and weighted PDF
        kl_div (float): Kullback-Leibler divergence between the PDF and weighted PDF
        js_div (float): Jensen-Shannon divergence between the PDF and weighted PDF
        """
        cos_sim = cosine_similarity(self.pdf, self.weighted_pdf)
        kl_div = kl_divergence(self.pdf, self.weighted_pdf)
        js_div = js_divergence(self.pdf, self.weighted_pdf)

        return cos_sim, kl_div, js_div
        """
        Find the MIDI numbers of the notes in the track where the weighted PDF is higher than the PDF

        Returns:
        long_notes (np array): MIDI numbers of the long notes
        """
        # Compare between the pdf and weighted pdf to find which indice of the weighted pdf is higher than the pdf
        long_notes = np.where(self.weighted_pdf > self.pdf)[0]
        # Find the midi number of the long notes
        long_notes = self.midi_range[long_notes]

        return long_notes        
    
def load_files(path, extension = ".xml"):
    """
    Load all files in a directory, including subdirectories, with a given extension
    
    Parameters:
    path (str): path to the directory
    extension (str): file extension to load, default is ".xml"

    Returns:
    files (list): list of file paths
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(extension)]:
            files.append(os.path.join(dirpath, filename))
    return files

def process_metadata(file_path):
    """
    Process metadata from a music21 file

    Parameters:
    file_path (str): path to the csv file

    Returns:
    metadata (pd dataframe): dataframe with metadata
    """
    # Read metadata csv file with pandas
    metadata = pd.read_csv(file_path, header=0)

    # Remove leading and trailing whitespaces from column names
    metadata.columns = metadata.columns.str.strip()

    # Change 'RECORDING_MBID' column to track_id and set it as the index
    metadata.rename(columns={'RECORDING_MBID': 'track_id'}, inplace=True)
    metadata.set_index('track_id', inplace=True)

    # Split the RECORDING_TITLE_TRANSLITERATED column into two columns, based on first ' ', first column is mizan and the rest is tab
    metadata['mizan'] = metadata['RECORDING_TITLE_TRANSLITERATED'].str.split(' ').str[0]
    metadata['tab'] = metadata['RECORDING_TITLE_TRANSLITERATED'].str.split(' ').str[1:].str.join(' ')

    # Remove leading and trailing whitespaces from the index, mizan and tab columns
    metadata.index = metadata.index.str.strip()
    metadata['mizan'] = metadata['mizan'].str.strip()
    metadata['tab'] = metadata['tab'].str.strip()

    # Convert columns to string
    metadata = metadata.astype(str)

    return metadata

def plot_histogram(note_sequence, title = None):
    """
    Plot a histogram of note counts in a note sequence, as a percentage of the total notes

    Parameters:
    note_sequence (list): list of notes
    title (str): title of the plot, default is None
    """
    # Get unique notes and their counts
    unique_notes, counts = np.unique(note_sequence, return_counts=True)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(unique_notes, counts/np.sum(counts))
    plt.xlabel('Note')
    plt.ylabel('Percentage of Total Notes')
    plt.title(title)
    plt.show()

#---------- Statistical functions ----------#

def chi_square_test(frequencies1, frequencies2, total_sample_size_1, total_sample_size_2):
    """
    Perform a Chi-square test of independence between two samples

    Parameters:
    frequencies1 (np array): frequencies of the first sample
    frequencies2 (np array): frequencies of the second sample
    total_sample_size_1 (int): total sample size of the first sample
    total_sample_size_2 (int): total sample size of the second sample

    Returns:
    chi2 (float): chi-square statistic
    p (float): p-value
    dof (int): degrees of freedom
    expected (np array): expected frequencies
    """
    # Convert frequencies to counts
    counts1 = frequencies1 * total_sample_size_1
    counts2 = frequencies2 * total_sample_size_2
    
    zero_indices1 = np.where(frequencies1 == 0)[0]  # Indices where frequencies1 is zero
    zero_indices2 = np.where(frequencies2 == 0)[0]  # Indices where frequencies2 is zero

    # Find the intersection of zero_indices1 and zero_indices2
    # These are the indices where both frequencies1 and frequencies2 are zero
    both_zero_indices = np.intersect1d(zero_indices1, zero_indices2)

    # Find the indices that are not in both_zero_indices
    # These are the indices where at least one of frequencies1 or frequencies2 is non-zero
    all_classes = np.setdiff1d(np.arange(0,35), both_zero_indices)

    # Create a contingency table from the two samples
    observed = np.zeros((len(all_classes), 2))
    for i, c in enumerate(all_classes):
        observed[i, 0] = counts1[c] if c < len(frequencies1) else 0
        observed[i, 1] = counts2[c] if c < len(frequencies2) else 0
    
    # Perform Chi-square test of independence
    chi2, p, dof, expected = chi2_contingency(observed)

    return chi2, p, dof, expected
    """
    Compute the cosine similarity between two probability distributions

    Parameters:
    prob_dist1 (np array): probability distribution 1
    prob_dist2 (np array): probability distribution 2

    Returns:
    cosine_sim (float): cosine similarity
    """
    dot_product = np.dot(prob_dist1, prob_dist2)
    norm_prob_dist1 = np.linalg.norm(prob_dist1)
    norm_prob_dist2 = np.linalg.norm(prob_dist2)
    cosine_sim = dot_product / (norm_prob_dist1 * norm_prob_dist2)
    return cosine_sim

def pearson_correlation(prob_dist1, prob_dist2):
    """
    Compute the Pearson correlation between two probability distributions

    Parameters:
    prob_dist1 (np array): probability distribution 1
    prob_dist2 (np array): probability distribution 2

    Returns:
    pearson_corr (float): Pearson correlation
    """
    mean1 = np.mean(prob_dist1)
    mean2 = np.mean(prob_dist2)
    cov   = np.mean((prob_dist1 - mean1) * (prob_dist2 - mean2))
    std1 = np.std(prob_dist1)
    std2 = np.std(prob_dist2)
    
    pearson_corr = cov / (std1 * std2)
    
    return pearson_corr

def cosine_similarity(p, q):
    """
    Compute the cosine similarity between two probability distributions

    Parameters:
    p (np array): probability distribution 1
    q (np array): probability distribution 2

    Returns:
    cosine_sim (float): cosine similarity
    """
    dot_product = np.dot(p, q)
    norm_p = np.linalg.norm(p)
    norm_q = np.linalg.norm(q)
    cosine_sim = dot_product / (norm_p * norm_q)
    return cosine_sim

def bhattacharyya_distance(p, q):
    """
    Compute the Bhattacharyya distance between two probability distributions

    Parameters:
    p (np array): probability distribution 1
    q (np array): probability distribution 2

    Returns:
    bhattacharyya_dist (float): Bhattacharyya distance
    """
    bhattacharyya_distance = -np.log(np.sum(np.sqrt(p * q)))
    return bhattacharyya_distance

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions

    Parameters:
    p (np array): probability distribution 1
    q (np array): probability distribution 2

    Returns:
    kl (float): Kullback-Leibler divergence
    """
    kl = np.sum(kl_div(p, q))
    return kl

def js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions

    Parameters:
    p (np array): probability distribution 1
    q (np array): probability distribution 2

    Returns:
    js_divergence (float): Jensen-Shannon divergence
    """
    m = 0.5 * (p + q)
    js_divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return js_divergence

def create_contingency_table(dictionary):
    """
    Create a contingency table from a dictionary

    Parameters:
    dictionary (dict): dictionary with keys in the format 'class1--class2' and values as counts

    Returns:
    contingency_table (np array): contingency table
    class_1_labels (list): unique class 1 labels
    class_2_labels (list): unique class 2 labels
    """
    # Extract unique class labels from keys
    class_1_labels = sorted(set([key.split('--')[0] for key in dictionary.keys()]))
    class_2_labels = sorted(set([key.split('--')[1] for key in dictionary.keys()]))
    
    # Create an empty contingency table
    contingency_table = np.zeros((len(class_1_labels), len(class_2_labels)))
    
    # Fill in the contingency table with values from the dictionary
    for key, value in dictionary.items():
        class_1, class_2 = key.split('--')
        i = class_1_labels.index(class_1)
        j = class_2_labels.index(class_2)
        contingency_table[i, j] = value
    
    return contingency_table, class_1_labels, class_2_labels

def plot_contingency_table(contingency_table, class_1_labels, class_2_labels, title):
    """
    Plot a contingency table

    Parameters:
    contingency_table (np array): contingency table
    class_1_labels (list): unique class 1 labels
    class_2_labels (list): unique class 2 labels
    title (str): title of the plot

    Returns:
    None

    NOTE: This function is provided to visualize a contingency table
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(contingency_table, cmap='rainbow', interpolation='nearest')

    plt.title(title)
    plt.xticks(np.arange(len(class_2_labels)), class_2_labels, rotation=45)
    plt.yticks(np.arange(len(class_1_labels)), class_1_labels)

    # Add text annotations
    for i in range(len(class_1_labels)):
        for j in range(len(class_2_labels)):
            plt.text(j, i, f'{contingency_table[i, j]:.2f}',  # Format numbers with two decimals
                     ha='center', va='center', color='white', fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.show()