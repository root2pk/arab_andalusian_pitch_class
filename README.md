# arab_andalusian_pitch_class
Companion repository for AMPLab assignment 2. Analysis of Arab Andalusian music scores to determine trends between pitch class histograms of scores across melodic and rhythmic frameworks.

## How to use the repository
`pre-process.py` is a script to pre-process the metadata and load Track objects associated with each score to `tracks.pkl`. The script has already been run and the files are available in the repository. The code is provided for reference.

`example_plot_pdfs` is an example notebook to compute the probability distribution functions of the notes in each score, and plot both unweighted (accounting for only note occurences) and weighted (accounting for note durations) pitch histograms.

`extract_non_weighted` is an example notebook to extract the nonweighted PDFs for each track, group them by tab and mizan, and plot graphs illustrating similarity metrics of scores with the same tab but different mizan.

`extract_weighted` is an example notebook that does the same as above but with weighted PDFs.

Utilizes [music21](https://web.mit.edu/music21/) to extract pitch information from [musicXML](https://www.musicxml.com/) scores.

Dataset is obtained from the wider [Dunya Arab Andalusian dataset](https://zenodo.org/records/1291776)

[1] M. Sordo, A. Chaachoo, and X. Serra, “Creating Corpora for Computational Research in Arab-Andalusian Music,” Proceedings of the 1st International Workshop on Digital Libraries for Musicology. ACM, Sep. 12, 2014. doi: 10.1145/2660168.2660182.
